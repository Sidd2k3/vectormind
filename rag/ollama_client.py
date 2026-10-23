"""Async client for a local Ollama server.

Ollama runs models locally and exposes them over HTTP (default port 11434).
Two endpoints are used:
    POST /api/embed      text(s) -> embedding vectors
    POST /api/generate   prompt  -> generated text (optionally streamed)

The client is async because LLM calls are slow and I/O-bound: while one
request waits on the model, the server keeps handling other requests.
"""

import json
from collections.abc import AsyncIterator

import httpx
import numpy as np


class OllamaError(RuntimeError):
    """Ollama is unreachable or returned an error. The message is shown to users."""


class OllamaClient:
    EMBED_BATCH_SIZE = 16

    def __init__(self, base_url: str, embed_model: str, gen_model: str,
                 temperature: float | None = None):
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.temperature = temperature
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            # Generation on a laptop CPU can take a while.
            timeout=httpx.Timeout(connect=3.0, read=300.0, write=30.0, pool=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def is_available(self) -> bool:
        try:
            r = await self._client.get("/api/tags", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        try:
            r = await self._client.get("/api/tags", timeout=2.0)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, ValueError, KeyError):
            return []

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            r = await self._client.post(path, json=payload)
        except httpx.ConnectError:
            raise OllamaError(f"Can't reach Ollama at {self.base_url}. Start it with: ollama serve") from None
        except httpx.TimeoutException:
            raise OllamaError("Ollama took too long to respond.") from None
        if r.status_code != 200:
            raise OllamaError(self._error_message(r.status_code, r.text, payload.get("model", "")))
        return r.json()

    @staticmethod
    def _error_message(status: int, body: str, model: str) -> str:
        try:
            detail = json.loads(body).get("error", body)
        except ValueError:
            detail = body
        if status == 404 or "not found" in str(detail).lower():
            return f"Model '{model}' isn't installed. Run: ollama pull {model}"
        return f"Ollama error ({status}): {detail}"

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed many texts, batched to keep each request small."""
        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self.EMBED_BATCH_SIZE):
            batch = texts[i:i + self.EMBED_BATCH_SIZE]
            data = await self._post("/api/embed", {"model": self.embed_model, "input": batch})
            embeddings = data.get("embeddings") or []
            if len(embeddings) != len(batch):
                raise OllamaError("Ollama returned the wrong number of embeddings.")
            vectors.extend(np.asarray(e, dtype=np.float32) for e in embeddings)
        return vectors

    def _generate_payload(self, prompt: str, stream: bool) -> dict:
        payload = {"model": self.gen_model, "prompt": prompt, "stream": stream}
        if self.temperature is not None:  # None = the model's default
            payload["options"] = {"temperature": self.temperature}
        return payload

    async def generate(self, prompt: str) -> str:
        data = await self._post("/api/generate", self._generate_payload(prompt, stream=False))
        return data.get("response", "").strip()

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield the answer piece by piece as the model produces it.
        Ollama streams newline-delimited JSON objects."""
        payload = self._generate_payload(prompt, stream=True)
        try:
            async with self._client.stream("POST", "/api/generate", json=payload) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode(errors="replace")
                    raise OllamaError(self._error_message(r.status_code, body, self.gen_model))
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    part = json.loads(line)
                    if "error" in part:
                        raise OllamaError(f"Ollama error: {part['error']}")
                    if part.get("response"):
                        yield part["response"]
                    if part.get("done"):
                        break
        except httpx.ConnectError:
            raise OllamaError(f"Can't reach Ollama at {self.base_url}. Start it with: ollama serve") from None
        except httpx.TimeoutException:
            raise OllamaError("Ollama took too long to respond.") from None
