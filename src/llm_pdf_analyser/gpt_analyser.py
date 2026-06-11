"""
OpenAI API calls with structured output (JSON Schema).
Implements retry with exponential backoff for 429 and 5xx errors.
"""

import json
import logging
import time

from openai import APIStatusError, OpenAI, RateLimitError

from .config import INITIAL_BACKOFF, MAX_RETRIES, MODEL
from .questions import QUESTIONS

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ── JSON Schema para o structured output ─────────────────────────────────────


def _build_json_schema() -> dict:
    """
    Builds the JSON Schema the model must conform to.
    One property per question:
      - single: a string matching one of the allowed options (or "Não aplicável")
      - multi:  an array of strings from the allowed options list
    """
    properties = {}
    required = []

    for q in QUESTIONS:
        q_id = q["id"]
        required.append(q_id)

        if q["tipo"] == "single":
            properties[q_id] = {
                "type": "string",
                "description": q["texto"],
                "enum": q["opcoes"] + ["Não aplicável"],
            }
        else:
            opcoes_validas = q["opcoes"] + (["Outro"] if q.get("tem_outro") else [])
            properties[q_id] = {
                "type": "array",
                "description": q["texto"],
                "items": {
                    "type": "string",
                    "enum": opcoes_validas,
                },
            }
            if q.get("tem_outro"):
                outro_key = f"{q_id}_outro_texto"
                properties[outro_key] = {
                    "type": "string",
                    "description": (
                        f"Texto livre para 'Outro' na pergunta '{q['texto']}'. "
                        "Deixar vazio se não aplicável."
                    ),
                }
                required.append(outro_key)

    return {
        "name": "analise_artigo",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


_JSON_SCHEMA = _build_json_schema()


_SYSTEM_PROMPT = """Você é um assistente especializado em análise de artigos científicos sobre agricultura de precisão e fruticultura.

Sua tarefa é ler o texto de um artigo científico e responder um conjunto de perguntas categóricas de múltipla escolha com base EXCLUSIVAMENTE no que está escrito no artigo.

Regras importantes:
- Marque apenas as opções que estão claramente mencionadas ou fortemente implícitas no texto.
- Para perguntas do tipo "single", escolha apenas UMA opção.
- Para perguntas do tipo "multi", selecione TODAS as opções aplicáveis.
- Se a informação não estiver disponível no texto, use "Não aplicável" (para single) ou lista vazia [] (para multi).
- Para campos "outro_texto", preencha com texto breve e objetivo apenas se "Outro" foi selecionado.
- Responda sempre em português."""


def _build_user_prompt(article_text: str) -> str:
    lines = ["# Texto do artigo\n", article_text, "\n\n# Perguntas\n"]
    for i, q in enumerate(QUESTIONS, 1):
        tipo_label = (
            "(marcar todas aplicáveis)" if q["tipo"] == "multi" else "(escolher uma)"
        )
        lines.append(f"{i}. [{q['id']}] {q['texto']} {tipo_label}")
        for opt in q["opcoes"]:
            lines.append(f"   - {opt}")
        if q.get("tem_outro"):
            lines.append("   - Outro (especificar)")
        lines.append("")
    return "\n".join(lines)


# Main API call
def analyse_article(article_text: str) -> dict:
    """
    Sends the article text to the GPT model and returns the parsed response dict.

    Raises:
        RuntimeError: if all retry attempts fail.
    """

    client = get_client()
    user_prompt = _build_user_prompt(article_text)

    backoff = INITIAL_BACKOFF
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "analise_artigo",
                        "schema": _JSON_SCHEMA["schema"],
                        "strict": True,
                    }
                },
            )
            raw = response.output_text
            if raw is None:
                raise ValueError("Model returned an empty response.")
            return json.loads(raw)

        except RateLimitError as exc:
            last_error = exc
            logger.warning(
                f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {backoff}s…"
            )
            time.sleep(backoff)
            backoff *= 2

        except APIStatusError as exc:
            if exc.status_code >= 500:
                last_error = exc
                logger.warning(
                    f"Server error {exc.status_code} (attempt {attempt}/{MAX_RETRIES}). Waiting {backoff}s…"
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                raise  # 4xx errors (except 429) are not retried

        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                f"Invalid JSON response (attempt {attempt}/{MAX_RETRIES}): {exc}"
            )
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")
