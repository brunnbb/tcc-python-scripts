"""
Checkpoint, extracted text cache, and CSV result management.

Checkpoint (checkpoint.json)
─────────────────────────────
Structure: { "filename.pdf": { "extract": status, "analyse": status } }

  extract:  "ok" | "pdf_sem_texto" | "erro_leitura"
  analyse:  "success" | "error" | "skipped" | null (null = not yet analysed)

Extracted text cache (data/out/texts/)
────────────────────────────────────────
One .txt file per PDF containing the cleaned text sent to the model.
Allows inspection of exactly what the model received and avoids re-extraction.

CSV (data/out/results.csv)
───────────────────────────
One row per article with binary columns for each answer option.
Written incrementally (append mode) — safe against interruptions.
"""

import csv
import json
import logging

from .config import CHECKPOINT_FILE, NORMALIZED_CSV, OUTPUT_CSV, OUTPUT_DIR, TEXTS_DIR
from .questions import QUESTIONS, _make_col_name, get_all_csv_columns

logger = logging.getLogger(__name__)


# ── Checkpoint ────────────────────────────────────────────────────────────────


def load_checkpoint() -> dict[str, dict]:
    """Load the existing checkpoint or return an empty dict."""
    if CHECKPOINT_FILE.exists():
        try:
            content = CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
            if content:
                return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("checkpoint.json is corrupted or empty — starting fresh.")
    return {}


def save_checkpoint(checkpoint: dict[str, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def checkpoint_entry(extract_status: str, analyse_status: str | None = None) -> dict:
    return {"extract": extract_status, "analyse": analyse_status}


# ── Extracted text cache ──────────────────────────────────────────────────────


def text_path(pdf_filename: str):
    """Returns the .txt cache path corresponding to a PDF filename."""
    from pathlib import Path

    stem = Path(pdf_filename).stem
    return TEXTS_DIR / f"{stem}.txt"


def save_extracted_text(pdf_filename: str, text: str) -> None:
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    text_path(pdf_filename).write_text(text, encoding="utf-8")


def load_extracted_text(pdf_filename: str) -> str | None:
    """Returns cached text, or None if not yet extracted."""
    p = text_path(pdf_filename)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


# ── CSV ───────────────────────────────────────────────────────────────────────

ALL_COLUMNS = get_all_csv_columns()


def init_csv() -> None:
    """Create the CSV with a header row, if it does not already exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
        logger.info(f"CSV created: {OUTPUT_CSV}")


def append_result(filename: str, status: str, gpt_response: dict | None) -> None:
    """
    Append one row to the CSV for the given article.

    - status "success":  fills binary columns from the GPT response.
    - any other status:  writes filename + status only; all other columns empty.
    """
    row: dict[str, str | int] = {col: "" for col in ALL_COLUMNS}
    row["arquivo"] = filename
    row["status"] = status

    if status == "success" and gpt_response is not None:
        _fill_row_from_gpt(row, gpt_response)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writerow(row)


def _fill_row_from_gpt(row: dict, gpt_response: dict) -> None:
    """Convert the GPT JSON response into binary columns in the CSV row."""
    for q in QUESTIONS:
        q_id = q["id"]
        value = gpt_response.get(q_id)

        if q["tipo"] == "single":
            for opt in q["opcoes"]:
                col = _make_col_name(q_id, opt)
                if col in row:
                    row[col] = 1 if (value == opt) else 0
        else:
            selected: list[str] = value if isinstance(value, list) else []
            for opt in q["opcoes"]:
                col = _make_col_name(q_id, opt)
                if col in row:
                    row[col] = 1 if (opt in selected) else 0
            if q.get("tem_outro"):
                outro_key = f"{q_id}_outro_texto"
                row[outro_key] = gpt_response.get(outro_key, "")


# ── Normalized CSV ───────────────────────────────────────────────────────────


def generate_normalized_csv() -> None:
    """
    Read results.csv and write results_normalized.csv with one column per question.

    - single questions: the selected option text (or "" if none).
    - multi questions:  comma-separated selected option texts; free-text "outro"
                        appended at the end when present.
    """
    if not OUTPUT_CSV.exists():
        logger.warning("results.csv not found — nothing to normalize.")
        return

    norm_cols = ["arquivo", "status"] + [q["id"] for q in QUESTIONS]

    with (
        open(OUTPUT_CSV, newline="", encoding="utf-8") as fin,
        open(NORMALIZED_CSV, "w", newline="", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=norm_cols)
        writer.writeheader()

        for row in reader:
            norm_row: dict[str, str] = {
                "arquivo": row.get("arquivo", ""),
                "status": row.get("status", ""),
            }

            for q in QUESTIONS:
                q_id = q["id"]

                if q["tipo"] == "single":
                    value = ""
                    for opt in q["opcoes"]:
                        if row.get(_make_col_name(q_id, opt)) == "1":
                            value = opt
                            break
                    norm_row[q_id] = value
                else:
                    selected = [
                        opt
                        for opt in q["opcoes"]
                        if row.get(_make_col_name(q_id, opt)) == "1"
                    ]
                    if q.get("tem_outro"):
                        outro = row.get(f"{q_id}_outro_texto", "").strip()
                        if outro:
                            selected.append(outro)
                    norm_row[q_id] = ", ".join(selected)

            writer.writerow(norm_row)

    logger.info(f"Normalized CSV written: {NORMALIZED_CSV}")


# ── Error log ─────────────────────────────────────────────────────────────────


def log_error(filename: str, reason: str) -> None:
    from .config import ERROR_LOG

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"{filename}\t{reason}\n")
