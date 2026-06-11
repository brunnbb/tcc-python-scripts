"""
Main pipeline orchestrator with two "independent", resumable phases.

  Phase 1 — Extraction
    Reads each PDF, extracts its text, and saves it to data/out/texts/<name>.txt

  Phase 2 — Analysis
    For each extracted .txt, calls the GPT model and saves results to the CSV.
    Cost is estimated from the real sizes of the extracted text files.

The checkpoint (data/out/checkpoint.json) tracks the status of each phase per
article, so any interruption resumes from the correct point.

Example Usage:
    uv run python -m src.llm_pdf_analyser.main               # run both phases
    uv run python -m src.llm_pdf_analyser.main --only-extract
    uv run python -m src.llm_pdf_analyser.main --only-analyse
    uv run python -m src.llm_pdf_analyser.main --dry-run --limit 5
"""

import argparse
import json
import logging
import sys
import time

from dotenv import load_dotenv
from tqdm import tqdm

from .config import (
    AVG_OUTPUT_TOKENS,
    MODEL,
    MODEL_PRICING,
    PDF_FOLDER,
    SLEEP_BETWEEN_CALLS,
    TEXTS_DIR,
)
from .gpt_analyser import analyse_article
from .pdf_extractor import extract_text
from .results_manager import (
    append_result,
    checkpoint_entry,
    init_csv,
    load_checkpoint,
    load_extracted_text,
    log_error,
    save_checkpoint,
    save_extracted_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4


def _prompt_overhead_tokens() -> int:
    """
    Fixed token overhead per API call: system prompt + question template.
    Conservative estimate; independent of article content.
    """
    return 1_500


def estimate_cost_from_files(pdfs: list, model: str) -> dict:
    """
    Estimate the API cost using real file sizes.

    Strategy:
    1. If the .txt already exists in data/out/texts/, use its size —
       that file contains exactly the text that will be sent to the model
       (already truncated to MAX_CHARS).
    2. If only the PDF exists (not yet extracted), use 15% of the PDF size
       as a proxy for extractable text. This overestimates slightly but is
       far more accurate than a fixed average.
    3. Tracks how many articles used each source for transparency.

    Returns a dict with: total_cost, input_cost, output_cost,
    total_input_tokens, total_output_tokens, avg_input_tokens,
    n_from_txt, n_from_pdf, n_missing.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        raise ValueError(f"Model '{model}' has no pricing entry in MODEL_PRICING.")

    total_input_tokens = 0
    overhead = _prompt_overhead_tokens()
    n_from_txt = n_from_pdf = n_missing = 0

    for pdf_path in pdfs:
        txt_path = TEXTS_DIR / (pdf_path.stem + ".txt")

        if txt_path.exists():
            chars = txt_path.stat().st_size
            tokens = chars // CHARS_PER_TOKEN
            n_from_txt += 1
        elif pdf_path.exists():
            # PDF not yet extracted — use file size as a rough proxy
            chars = pdf_path.stat().st_size
            tokens = int(chars * 0.15) // CHARS_PER_TOKEN
            n_from_pdf += 1
        else:
            tokens = 0
            n_missing += 1

        total_input_tokens += tokens + overhead

    total_output_tokens = len(pdfs) * AVG_OUTPUT_TOKENS

    input_cost = (total_input_tokens / 1_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "total_cost": total_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_input_tokens": total_input_tokens // max(len(pdfs), 1),
        "n_from_txt": n_from_txt,
        "n_from_pdf": n_from_pdf,
        "n_missing": n_missing,
    }


def print_cost_table(pdfs: list) -> None:
    """Print a comparison table of estimated cost across all registered models."""
    print(f"\n  {'Model':<14} {'Input':>8} {'Output':>8} {'Total':>8}  Token source")
    print(f"  {'─' * 14} {'─' * 8} {'─' * 8} {'─' * 8}  {'─' * 24}")

    for m in MODEL_PRICING:
        est = estimate_cost_from_files(pdfs, m)
        marker = " ◄ default" if m == MODEL else ""
        source = (
            f"{est['n_from_txt']} txt + {est['n_from_pdf']} pdf"
            if (est["n_from_txt"] + est["n_from_pdf"]) > 0
            else "estimated"
        )
        print(
            f"  {m:<14}  ${est['input_cost']:>6.2f}  ${est['output_cost']:>6.2f}"
            f"  ${est['total_cost']:>6.2f}  {source}{marker}"
        )

    est_default = estimate_cost_from_files(pdfs, MODEL)
    print(
        f"\n  Average tokens per article: ~{est_default['avg_input_tokens']:,} input + {AVG_OUTPUT_TOKENS} output"
    )


# ── Phase 1: Extraction ───────────────────────────────────────────────────────


def phase_extract(pdfs: list, checkpoint: dict, dry_run: bool) -> dict:
    """
    Extract text from all PDFs that have not yet been processed.
    Saves .txt files to data/out/texts/ and updates the checkpoint.
    Returns the updated checkpoint.
    """
    pending = [
        p
        for p in pdfs
        if checkpoint.get(p.name, {}).get("extract") not in ("ok", "pdf_sem_texto")
    ]

    print(f"\n{'─' * 60}")
    print("  Text extraction")
    print(f"  PDFs to extract : {len(pending)} of {len(pdfs)}")
    print(f"{'─' * 60}\n")

    if not pending:
        print("  All PDFs have already been extracted.\n")
        return checkpoint

    n_ok = n_no_text = n_error = 0

    for pdf_path in tqdm(pending, desc="Extracting PDFs", unit="pdf"):
        filename = pdf_path.name
        text, status = extract_text(pdf_path)

        if dry_run:
            print(f"  {filename}: {status} ({len(text)} chars)")
            continue

        if status == "ok":
            save_extracted_text(filename, text)
            n_ok += 1
        elif status == "pdf_sem_texto":
            # Save partial text anyway — useful for debugging
            save_extracted_text(filename, text)
            log_error(filename, "pdf_sem_texto")
            n_no_text += 1
        else:
            log_error(filename, status)
            n_error += 1

        checkpoint[filename] = checkpoint_entry(
            extract_status=status,
            analyse_status=checkpoint.get(filename, {}).get("analyse"),
        )
        save_checkpoint(checkpoint)

    if not dry_run:
        print(f"\n  ✓ Extracted with text : {n_ok}")
        print(f"  ○ No extractable text : {n_no_text}")
        print(f"  ✗ Read errors         : {n_error}\n")

    return checkpoint


# ── Phase 2: GPT Analysis ─────────────────────────────────────────────────────


def phase_analyse(pdfs: list, checkpoint: dict, dry_run: bool) -> None:
    """
    Analyse with GPT all articles that have been extracted but not yet analysed.
    Saves results to the CSV and updates the checkpoint.
    """
    pending = [
        p
        for p in pdfs
        if checkpoint.get(p.name, {}).get("extract") == "ok"
        and checkpoint.get(p.name, {}).get("analyse") != "success"
    ]

    n_no_text = sum(
        1 for p in pdfs if checkpoint.get(p.name, {}).get("extract") == "pdf_sem_texto"
    )

    print(f"\n{'─' * 60}")
    print(f"  LLM {MODEL} analysis")
    print(f"  Articles to analyse : {len(pending)} of {len(pdfs)}")
    print(f"  PDFs without text   : {n_no_text} (skipped)")

    # Cost estimate based on real sizes of pending articles
    print_cost_table(pending)
    print(f"{'─' * 60}\n")

    if not pending:
        print("  All articles have already been analysed.\n")
        return

    if not dry_run:
        confirm = input("  Confirm analysis? [y/N] ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return
        init_csv()
        # Register no-text PDFs in the CSV (only on first run)
        for p in pdfs:
            fname = p.name
            if checkpoint.get(fname, {}).get("extract") == "pdf_sem_texto":
                if checkpoint.get(fname, {}).get("analyse") is None:
                    append_result(fname, "pdf_sem_texto", None)
                    checkpoint[fname] = checkpoint_entry("pdf_sem_texto", "skipped")
        save_checkpoint(checkpoint)

    n_success = n_errors = 0

    for pdf_path in tqdm(pending, desc="Analysing with GPT", unit="article"):
        filename = pdf_path.name

        text = load_extracted_text(filename)
        if text is None:
            logger.error(f"Cached text not found for {filename}")
            n_errors += 1
            continue

        try:
            gpt_response = analyse_article(text)

            if dry_run:
                print(f"\n--- {filename} ---")
                print(json.dumps(gpt_response, ensure_ascii=False, indent=2))
            else:
                append_result(filename, "success", gpt_response)
                checkpoint[filename] = checkpoint_entry("ok", "success")
                save_checkpoint(checkpoint)

            n_success += 1

        except Exception as exc:
            logger.error(f"Error analysing {filename}: {exc}")
            n_errors += 1
            if not dry_run:
                append_result(filename, "error", None)
                checkpoint[filename] = checkpoint_entry("ok", "error")
                save_checkpoint(checkpoint)
                log_error(filename, str(exc))

        if not dry_run:
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\n  ✓ Success : {n_success}")
    print(f"  ✗ Errors  : {n_errors}\n")


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run(
    dry_run: bool = False,
    limit: int | None = None,
    only_extract: bool = False,
    only_analyse: bool = False,
) -> None:
    load_dotenv()

    if not PDF_FOLDER.exists():
        logger.error(f"PDF folder not found: {PDF_FOLDER}")
        sys.exit(1)

    pdfs = sorted(PDF_FOLDER.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    if not pdfs:
        logger.warning("No PDFs found in data/in/pdfs/. Exiting.")
        return

    checkpoint = load_checkpoint()

    print(f"\n{'=' * 60}")
    print(f"  PDFs found : {len(pdfs)}")
    if limit:
        print(f"  (limited to {limit} by --limit)")
    print(f"{'=' * 60}")

    if not only_analyse:
        checkpoint = phase_extract(pdfs, checkpoint, dry_run)

    if not only_extract:
        phase_analyse(pdfs, checkpoint, dry_run)

    if not (dry_run or only_extract):
        from .config import OUTPUT_CSV

        print(f"{'=' * 60}")
        print(f"  Output CSV : {OUTPUT_CSV}")
        print(f"{'=' * 60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF analysis pipeline using GPT (extraction + analysis)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving anything.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N PDFs.",
    )
    parser.add_argument(
        "--only-extract",
        action="store_true",
        help="Run Phase 1 only (text extraction).",
    )
    parser.add_argument(
        "--only-analyse",
        action="store_true",
        help="Run Phase 2 only (GPT analysis). Requires prior extraction.",
    )
    args = parser.parse_args()

    if args.only_extract and args.only_analyse:
        parser.error("--only-extract and --only-analyse are mutually exclusive.")

    run(
        dry_run=args.dry_run,
        limit=args.limit,
        only_extract=args.only_extract,
        only_analyse=args.only_analyse,
    )


if __name__ == "__main__":
    main()
