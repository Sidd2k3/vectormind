"""FastAPI server: REST API for both databases, plus the web UI.

Run:  python -m app          (or: uvicorn app.main:app --port 8080)
Docs: http://localhost:8080/docs
"""

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag.ollama_client import OllamaClient, OllamaError
from rag.pipeline import RAGPipeline
from vectordb.database import ALGORITHMS, DocumentDB, VectorDB
from vectordb.demo_data import CATEGORIES, DIMS, baseline_vector, demo_vectors, encode_text
from vectordb.distance import METRICS
from vectordb.pca import PCA2D

from .config import Settings
from .schemas import DocInsertRequest, InsertItemRequest, QuestionRequest

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, llm=None) -> FastAPI:
    settings = settings or Settings()
    llm = llm or OllamaClient(settings.ollama_url, settings.embed_model, settings.gen_model)

    demo_db = VectorDB(DIMS, hnsw_m=16)
    for text, category, vector in demo_vectors():
        demo_db.insert(text, category, vector)

    doc_db = DocumentDB(settings.data_dir if settings.persist else None, settings.embed_model)
    rag = RAGPipeline(doc_db, llm, settings.chunk_words, settings.overlap_words, settings.max_distance)

    # ── document markers on the demo map ──────────────────────────────
    # Every document also gets one 16D marker in the demo DB (category "doc"),
    # encoded with the keyword encoder, so it appears on the scatter plot and
    # can be highlighted when RAG retrieves it. This is for visualization only:
    # RAG always searches the real 768D document index.
    doc_markers: dict[str, int] = {}  # document title -> demo item id

    def base_title(chunk_title: str) -> str:
        return re.sub(r" \[\d+/\d+\]$", "", chunk_title)

    def add_marker(title: str, text: str) -> int:
        if title in doc_markers:
            demo_db.remove(doc_markers[title])
        vector = encode_text(f"{title} {text}")
        doc_markers[title] = demo_db.insert(title, "doc", vector if vector is not None else baseline_vector())
        return doc_markers[title]

    # Rebuild markers for documents loaded from disk.
    by_title: dict[str, list[str]] = {}
    for chunk in doc_db.all():
        by_title.setdefault(base_title(chunk.title), []).append(chunk.text)
    for title, texts in by_title.items():
        add_marker(title, " ".join(texts))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"VectorMind  →  http://{settings.host}:{settings.port}")
        print(f"  demo: {len(demo_db)} vectors, {DIMS}D | documents: {len(doc_db)} chunks")
        up = await llm.is_available()
        print(f"  Ollama: {'online' if up else 'OFFLINE (run: ollama serve)'}"
              f" | embed: {llm.embed_model} | generate: {llm.gen_model}")
        yield
        await llm.aclose()

    app = FastAPI(title="VectorMind",
                  description="Vector database from scratch (Brute Force, KD-Tree, HNSW) + local RAG.",
                  version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
    app.state.demo_db, app.state.doc_db, app.state.rag = demo_db, doc_db, rag

    # Domain errors become clean HTTP errors.
    @app.exception_handler(ValueError)
    async def bad_input(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(OllamaError)
    async def ollama_down(_: Request, exc: OllamaError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    # ── helpers ───────────────────────────────────────────────────────

    def parse_query(v: str | None, q: str | None):
        """A demo query is either a raw vector (?v=0.1,0.2,...) or text (?q=guitar)."""
        if v:
            try:
                return [float(x) for x in v.split(",")]
            except ValueError:
                raise ValueError("v must be comma-separated numbers") from None
        if q:
            # Unknown words give the neutral vector, so a search always runs.
            vector = encode_text(q)
            return vector if vector is not None else baseline_vector()
        raise ValueError("Provide a text query (q) or a vector (v)")

    def item_json(item, pca, distance=None):
        x, y = pca.project(item.vector)
        d = {"id": item.id, "text": item.text, "category": item.category, "x": x, "y": y}
        if distance is not None:
            d["distance"] = round(distance, 6)
        return d

    # ── demo vector endpoints ─────────────────────────────────────────

    @app.get("/search", tags=["demo"])
    def search(v: str | None = None, q: str | None = None, k: int = Query(5, ge=1, le=50),
               metric: str = "cosine", algo: str = "hnsw"):
        query = parse_query(v, q)
        hits, latency = demo_db.search(query, k, metric, algo)
        pca = PCA2D([i.vector for i in demo_db.items()])
        qx, qy = pca.project(query)
        return {"results": [item_json(item, pca, d) for d, item in hits],
                "latency_us": round(latency, 2), "algo": algo, "metric": metric, "k": k,
                "query": {"vector": [round(float(x), 4) for x in query], "x": qx, "y": qy}}

    @app.post("/insert", status_code=201, tags=["demo"])
    def insert(body: InsertItemRequest):
        if body.embedding is not None:
            vector = body.embedding
        else:
            # The category name counts too, so an item always leans toward its topic.
            vector = encode_text(f"{body.text} {body.category}")
            if vector is None:
                vector = baseline_vector()
        return {"id": demo_db.insert(body.text, body.category, vector)}

    @app.delete("/delete/{item_id}", tags=["demo"])
    def delete(item_id: int):
        if not demo_db.remove(item_id):
            raise HTTPException(404, f"No demo item with id {item_id}")
        for title, marker in list(doc_markers.items()):
            if marker == item_id:
                del doc_markers[title]
        return {"ok": True}

    @app.get("/items", tags=["demo"])
    def items():
        all_items = demo_db.items()
        pca = PCA2D([i.vector for i in all_items])
        out = []
        for item in all_items:
            d = item_json(item, pca)
            d["embedding"] = [round(float(x), 4) for x in item.vector]
            out.append(d)
        return {"items": out, "dims": DIMS}

    @app.get("/benchmark", tags=["demo"])
    def benchmark(v: str | None = None, q: str | None = None, k: int = Query(5, ge=1, le=50),
                  metric: str = "cosine"):
        return demo_db.benchmark(parse_query(v, q), k, metric)

    @app.get("/hnsw-info", tags=["demo"])
    def hnsw_info(metric: str = "cosine"):
        info = demo_db.hnsw_info(metric)
        for node in info["nodes"]:
            item = demo_db.get(node["id"])
            node["text"], node["category"] = item.text, item.category
        return info

    @app.get("/stats", tags=["demo"])
    def stats():
        return {"count": len(demo_db), "dims": DIMS, "algorithms": list(ALGORITHMS),
                "metrics": list(METRICS), "categories": list(CATEGORIES)}

    # ── document + RAG endpoints ──────────────────────────────────────

    def with_markers(contexts: list[dict]) -> list[dict]:
        """Attach the demo-map marker id of each retrieved chunk's document."""
        for c in contexts:
            c["marker_id"] = doc_markers.get(base_title(c["title"]))
        return contexts

    @app.post("/doc/insert", status_code=201, tags=["documents"])
    async def doc_insert(body: DocInsertRequest):
        title = body.title.strip()
        result = await rag.ingest(title, body.text)
        result["marker_id"] = add_marker(title, body.text)
        return result

    @app.get("/doc/list", tags=["documents"])
    def doc_list():
        docs = []
        for c in doc_db.all():
            preview = c.text[:140] + ("…" if len(c.text) > 140 else "")
            docs.append({"id": c.id, "title": c.title, "preview": preview, "words": len(c.text.split())})
        return {"documents": docs, "count": len(docs), "dims": doc_db.dims}

    @app.delete("/doc/delete/{chunk_id}", tags=["documents"])
    def doc_delete(chunk_id: int):
        chunk = next((c for c in doc_db.all() if c.id == chunk_id), None)
        if chunk is None or not doc_db.remove(chunk_id):
            raise HTTPException(404, f"No document chunk with id {chunk_id}")
        # Last chunk of a document gone: remove its marker from the map too.
        title = base_title(chunk.title)
        if not any(base_title(c.title) == title for c in doc_db.all()) and title in doc_markers:
            demo_db.remove(doc_markers.pop(title))
        return {"ok": True}

    @app.post("/doc/search", tags=["rag"])
    async def doc_search(body: QuestionRequest):
        """Retrieval only, no LLM. Fast, so the UI can show sources immediately."""
        contexts = await rag.retrieve(body.question, body.k)
        return {"contexts": with_markers([c.to_dict() for c in contexts])}

    @app.post("/doc/ask", tags=["rag"])
    async def doc_ask(body: QuestionRequest):
        out = await rag.ask(body.question, body.k)
        with_markers(out["contexts"])
        return out

    @app.post("/doc/ask/stream", tags=["rag"])
    async def doc_ask_stream(body: QuestionRequest):
        """Same as /doc/ask, but streams newline-delimited JSON events
        (contexts first, then answer tokens as they are generated)."""
        async def events():
            async for event in rag.ask_stream(body.question, body.k):
                if event["type"] == "contexts":
                    with_markers(event["contexts"])
                yield json.dumps(event) + "\n"
        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.get("/status", tags=["rag"])
    async def status():
        installed = await llm.list_models()
        names = {m.split(":")[0] for m in installed} | set(installed)
        return {"ollama_available": bool(installed) or await llm.is_available(),
                "embed_model": llm.embed_model, "gen_model": llm.gen_model,
                "embed_model_installed": llm.embed_model in names,
                "gen_model_installed": llm.gen_model in names,
                "doc_count": len(doc_db), "doc_dims": doc_db.dims,
                "demo_count": len(demo_db), "demo_dims": DIMS}

    # ── web UI ────────────────────────────────────────────────────────

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
