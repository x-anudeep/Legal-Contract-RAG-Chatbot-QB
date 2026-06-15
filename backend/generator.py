import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()
load_dotenv(Path(__file__).with_name(".env"))


SYSTEM_PROMPT = """You are a legal contract analyst assistant.

Answer strictly from the provided contract excerpts. If the relevant clause is not present,
say: "This clause was not found in the retrieved contract sections."
Always cite the contract and clause type used for the answer."""


def _format_source(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "contract": chunk.get("contract_name"),
        "contract_name": chunk.get("contract_name"),
        "clause_categories": chunk.get("clause_categories", []),
        "text_preview": chunk.get("text_preview") or chunk.get("text", "")[:300],
    }


def _extractive_generate(query: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "This clause was not found in the retrieved contract sections."

    best = chunks[0]
    clause_label = ", ".join(best.get("clause_categories", [])) or "Unlabeled"
    answers = [str(answer).strip() for answer in best.get("answers", []) if str(answer).strip()]
    lead = answers[0] if answers else "Based on the retrieved excerpt"

    return (
        f"{lead}. Source: {best.get('contract_name')} ({clause_label}). "
        f"Relevant text: \"{best.get('text', '')}\""
    )


def _build_context(chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        clause_label = ", ".join(chunk.get("clause_categories", [])) or "Unlabeled"
        context_blocks.append(
            f"[Excerpt {index}]\n"
            f"Contract: {chunk.get('contract_name')}\n"
            f"Clause type: {clause_label}\n"
            f"Text: {chunk.get('text')}"
        )
    return "\n\n---\n\n".join(context_blocks)


def _build_user_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    return f"Contract excerpts:\n\n{_build_context(chunks)}\n\nQuestion: {query}"


def _gemini_generate(query: str, chunks: list[dict[str, Any]]) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to backend/.env or export it in your shell."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        contents=_build_user_prompt(query, chunks),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )
    return response.text or ""


def _openai_generate(query: str, chunks: list[dict[str, Any]]) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query, chunks)},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def generate(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    mode = os.getenv("GENERATOR_MODE", "extractive").lower()
    if mode == "gemini":
        answer = _gemini_generate(query, chunks)
    elif mode == "openai":
        answer = _openai_generate(query, chunks)
    else:
        answer = _extractive_generate(query, chunks)

    return {
        "answer": answer,
        "sources": [_format_source(chunk) for chunk in chunks],
    }
