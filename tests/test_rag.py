import asyncio

import httpx
import numpy as np
import pytest

from rag.chunker import chunk_text
from rag.ollama_client import OllamaClient, OllamaError
from rag.pipeline import RAGPipeline, build_prompt
from tests.fakes import FakeOllama
from vectordb.database import DocumentDB


def run(coro):
    return asyncio.run(coro)


# ── chunker ───────────────────────────────────────────────────────────

def test_chunker_windows_and_overlap():
    words = [f"w{i}" for i in range(600)]
    chunks = chunk_text(" ".join(words), 250, 30)
    assert len(chunks) == 3
    split = [c.split() for c in chunks]
    assert split[0][0] == "w0" and split[0][-1] == "w249"
    assert split[1][0] == "w220" and split[1][-1] == "w469"
    assert split[2][0] == "w440" and split[2][-1] == "w599"
    assert split[0][-30:] == split[1][:30]  # the overlap


def test_chunker_short_and_empty():
    assert chunk_text("  hello\nworld  ") == ["hello\nworld"]
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("a b", 10, 10)


# ── pipeline (with a fake Ollama) ─────────────────────────────────────

def make_pipeline():
    return RAGPipeline(DocumentDB(None), FakeOllama())


def test_ingest_and_ask_uses_context():
    rag = make_pipeline()
    run(rag.ingest("OS notes", "A deadlock happens when processes wait on each other forever."))
    run(rag.ingest("Cooking", "Bake the bread at 220 degrees for thirty minutes."))
    out = run(rag.ask("what is a deadlock", 3))
    assert out["contexts"][0]["title"] == "OS notes"
    assert all(c["title"] != "Cooking" for c in out["contexts"])  # filtered by threshold
    assert out["answer"].startswith("ANSWER:")
    prompt = rag.llm.last_prompt
    assert "[1] OS notes:" in prompt and "general knowledge" in prompt


def test_no_relevant_context_falls_back_to_general_knowledge():
    rag = make_pipeline()
    run(rag.ingest("Cooking", "Bake the bread at 220 degrees."))
    out = run(rag.ask("deadlock processes", 3))
    assert out["contexts"] == []          # nothing passed the relevance threshold
    assert out["answer"].startswith("ANSWER:")  # the LLM still answers
    assert "Context:\n(none)" in rag.llm.last_prompt


def test_nomic_prefixes_used():
    rag = make_pipeline()
    run(rag.ingest("T", "deadlock text"))
    run(rag.retrieve("deadlock", 1))
    assert rag.llm.embedded[0].startswith("search_document: ")
    assert rag.llm.embedded[1].startswith("search_query: ")


def test_long_document_chunk_titles():
    rag = make_pipeline()
    out = run(rag.ingest("Big", " ".join(["deadlock"] * 600)))
    assert out["chunks"] == 3
    assert [c.title for c in rag.docs.all()] == ["Big [1/3]", "Big [2/3]", "Big [3/3]"]


def test_stream_events():
    rag = make_pipeline()
    run(rag.ingest("OS", "deadlock processes waiting"))

    async def collect():
        return [e async for e in rag.ask_stream("deadlock", 3)]

    events = run(collect())
    assert events[0]["type"] == "contexts"
    assert events[-1]["type"] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text.startswith("ANSWER:")


def test_build_prompt_numbers_sources():
    from rag.pipeline import RetrievedChunk
    from vectordb.database import Chunk
    ctx = [RetrievedChunk(1, 0.1, Chunk(5, "A", "alpha", np.zeros(2))),
           RetrievedChunk(2, 0.2, Chunk(6, "B", "beta", np.zeros(2)))]
    p = build_prompt("  q?  ", ctx)
    assert "[1] A:\nalpha" in p and "[2] B:\nbeta" in p and "Question: q?" in p


# ── real client against a mocked HTTP server ──────────────────────────

def client_with(handler):
    c = OllamaClient("http://ollama.test", "nomic-embed-text", "llama3.2")
    c._client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return c


def test_client_embed_batches_and_generate():
    calls = []

    def handler(request):
        import json
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * len(body["input"])})
        if request.url.path == "/api/generate" and not body["stream"]:
            return httpx.Response(200, json={"response": " hi ", "done": True})
        lines = '{"response":"he","done":false}\n{"response":"llo","done":false}\n{"done":true}\n'
        return httpx.Response(200, text=lines)

    c = client_with(handler)
    vecs = run(c.embed([f"t{i}" for i in range(20)]))
    assert len(vecs) == 20 and [len(b["input"]) for _, b in calls] == [16, 4]
    assert run(c.generate("p")) == "hi"

    async def stream():
        return "".join([p async for p in c.generate_stream("p")])
    assert run(stream()) == "hello"


def test_client_errors():
    c = client_with(lambda r: httpx.Response(404, json={"error": "model 'llama3.2' not found"}))
    with pytest.raises(OllamaError, match="ollama pull"):
        run(c.generate("p"))

    def refuse(request):
        raise httpx.ConnectError("refused")
    c = client_with(refuse)
    with pytest.raises(OllamaError, match="ollama serve"):
        run(c.embed(["x"]))
    assert run(c.is_available()) is False
