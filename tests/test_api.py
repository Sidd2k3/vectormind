import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.fakes import FakeOllama


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, persist=True)
    with TestClient(create_app(settings, llm=FakeOllama())) as c:
        yield c


def test_ui_and_docs_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "VectorMind" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/docs").status_code == 200


def test_search_by_text_and_vector(client):
    r = client.get("/search", params={"q": "guitar song", "k": 3, "algo": "kdtree"}).json()
    assert len(r["results"]) == 3 and r["results"][0]["category"] == "music"
    assert {"x", "y", "vector"} <= set(r["query"])

    vec = ",".join(str(x) for x in r["query"]["vector"])
    r2 = client.get("/search", params={"v": vec, "k": 3, "metric": "euclidean"}).json()
    assert r2["results"][0]["category"] == "music"


def test_search_errors(client):
    assert client.get("/search", params={"q": "xyzzy"}).status_code == 200  # neutral vector
    assert client.get("/search", params={"v": "1,2,3"}).status_code == 400
    assert client.get("/search", params={"q": "music", "metric": "cosin"}).status_code == 400
    assert client.get("/search", params={"q": "music", "k": 0}).status_code == 422


def test_insert_delete_items(client):
    r = client.post("/insert", json={"text": "Kubernetes runs docker containers in the cloud", "category": "tech"})
    assert r.status_code == 201
    new_id = r.json()["id"]
    items = client.get("/items").json()["items"]
    assert len(items) == 21 and len(items[0]["embedding"]) == 16
    assert client.delete(f"/delete/{new_id}").json() == {"ok": True}
    assert client.delete(f"/delete/{new_id}").status_code == 404
    # Unknown words still insert: the category counts, else a neutral vector is used.
    assert client.post("/insert", json={"text": "zzz", "category": "space"}).status_code == 201
    assert client.post("/insert", json={"text": "zzz", "category": "x"}).status_code == 201
    assert client.post("/insert", json={"text": ""}).status_code == 422


def test_documents_appear_on_the_demo_map(tmp_path):
    settings = Settings(data_dir=tmp_path, persist=True)
    with TestClient(create_app(settings, llm=FakeOllama())) as c:
        r = c.post("/doc/insert", json={"title": "OS", "text": "deadlock processes waiting " * 200}).json()
        assert r["chunks"] > 1
        markers = [i for i in c.get("/items").json()["items"] if i["category"] == "doc"]
        assert [m["id"] for m in markers] == [r["marker_id"]] and markers[0]["text"] == "OS"

        ctx = c.post("/doc/search", json={"question": "deadlock"}).json()["contexts"]
        assert ctx and all(x["marker_id"] == r["marker_id"] for x in ctx)

    # After a restart the marker is rebuilt from the saved documents.
    with TestClient(create_app(settings, llm=FakeOllama())) as c:
        docs = c.get("/doc/list").json()["documents"]
        assert sum(i["category"] == "doc" for i in c.get("/items").json()["items"]) == 1
        # Deleting every chunk of the document removes its marker.
        for d in docs:
            c.delete(f"/doc/delete/{d['id']}")
        assert not any(i["category"] == "doc" for i in c.get("/items").json()["items"])


def test_benchmark_hnsw_info_stats(client):
    b = client.get("/benchmark", params={"q": "planet"}).json()
    assert set(b["timings_us"]) == {"bruteforce", "kdtree", "hnsw"}
    assert all(v == 1.0 for v in b["recall"].values())
    info = client.get("/hnsw-info").json()
    assert info["node_count"] == 20 and info["nodes"][0]["text"]
    assert client.get("/stats").json()["dims"] == 16


def test_document_flow_and_rag(client):
    r = client.post("/doc/insert", json={"title": "OS", "text": "A deadlock is when processes wait forever."})
    assert r.status_code == 201 and r.json()["chunks"] == 1
    client.post("/doc/insert", json={"title": "Bread", "text": "Bake at 220 degrees for 30 minutes."})
    docs = client.get("/doc/list").json()
    assert docs["count"] == 2

    s = client.post("/doc/search", json={"question": "deadlock"}).json()
    assert s["contexts"][0]["title"] == "OS"

    a = client.post("/doc/ask", json={"question": "what is a deadlock", "k": 3}).json()
    assert a["answer"].startswith("ANSWER") and a["contexts"][0]["text"]

    with client.stream("POST", "/doc/ask/stream", json={"question": "deadlock"}) as resp:
        events = [json.loads(line) for line in resp.iter_lines() if line]
    assert events[0]["type"] == "contexts" and events[-1]["type"] == "done"

    chunk_id = docs["documents"][0]["id"]
    assert client.delete(f"/doc/delete/{chunk_id}").json() == {"ok": True}
    assert client.delete(f"/doc/delete/{chunk_id}").status_code == 404


def test_documents_persist_across_restart(tmp_path):
    settings = Settings(data_dir=tmp_path, persist=True)
    with TestClient(create_app(settings, llm=FakeOllama())) as c:
        c.post("/doc/insert", json={"title": "OS", "text": "deadlock processes"})
    with TestClient(create_app(settings, llm=FakeOllama())) as c:
        assert c.get("/doc/list").json()["count"] == 1


def test_status(client):
    s = client.get("/status").json()
    assert s["ollama_available"] and s["embed_model_installed"] and s["gen_model_installed"]


def test_ollama_offline_returns_503(tmp_path):
    from rag.ollama_client import OllamaError

    class Down(FakeOllama):
        async def embed(self, texts):
            raise OllamaError("Can't reach Ollama. Start it with: ollama serve")

    with TestClient(create_app(Settings(data_dir=tmp_path), llm=Down(available=False))) as c:
        r = c.post("/doc/insert", json={"title": "x", "text": "y"})
        assert r.status_code == 503 and "ollama serve" in r.json()["detail"]
        assert c.get("/status").json()["ollama_available"] is False
