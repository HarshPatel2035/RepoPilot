"""Turn retrieved chunks into a grounded, cited answer."""

from typing import Any, Protocol

SYSTEM_PROMPT = """You are RepoPilot, an expert on one code repository.
Answer using ONLY the numbered context chunks from that repository.
Cite every claim with the chunk label it came from, for example [src/app.py:10-40].
If the context does not contain the answer, say you could not find it in the repository.
Do not invent files, functions or behaviour."""


class Answerer(Protocol):
    def answer(self, question: str, hits: list[dict[str, Any]]) -> str: ...


def build_messages(question: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    if hits:
        context = "\n\n".join(
            f"### Chunk {n} [{h['path']}:{h['start_line']}-{h['end_line']}]\n{h['content']}"
            for n, h in enumerate(hits, start=1)
        )
    else:
        context = "(no matching code was found in the repository)"
    user = f"Context:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class GroqAnswerer:
    def __init__(self, api_key: str, model: str, timeout_seconds: float = 60.0) -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key, timeout=timeout_seconds)
        self._model = model

    def answer(self, question: str, hits: list[dict[str, Any]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model, messages=build_messages(question, hits), temperature=0.1
        )
        return response.choices[0].message.content or ""
