"""The RAG pipeline.

Indexing:  text -> chunks -> embeddings -> DocumentDB (HNSW, cosine)
Querying:  question -> embedding -> top-k chunks (filtered by distance)
           -> prompt with those chunks -> LLM -> answer + sources

The LLM uses the retrieved chunks when they are relevant, and falls back
to its own general knowledge when they aren't (or when no document is
close enough to the question).
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from vectordb.database import Chunk, DocumentDB

from .chunker import chunk_text
from .ollama_client import OllamaClient, OllamaError

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question directly and clearly.
If the context below contains relevant information, base your answer on it.
If it doesn't, answer from your own general knowledge.
Never refer to "the context", "the documents" or "the provided text" in your answer, and never say that the context lacks something. Just answer naturally.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RetrievedChunk:
    rank: int
    distance: float
    chunk: Chunk

    def to_dict(self, include_text: bool = True) -> dict:
        d = {"rank": self.rank, "id": self.chunk.id, "title": self.chunk.title,
             "distance": round(self.distance, 4)}
        if include_text:
            d["text"] = self.chunk.text
        return d


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.rank}] {c.chunk.title}:\n{c.chunk.text}" for c in contexts)
    return PROMPT_TEMPLATE.format(context=context or "(none)", question=question.strip())


class RAGPipeline:
    def __init__(self, docs: DocumentDB, llm: OllamaClient, chunk_words: int = 250,
                 overlap_words: int = 30, max_distance: float = 0.7):
        self.docs = docs
        self.llm = llm
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.max_distance = max_distance
        # nomic-embed-text was trained with task prefixes; using them makes
        # questions and passages line up better in the vector space.
        nomic = "nomic" in llm.embed_model.lower()
        self._doc_prefix = "search_document: " if nomic else ""
        self._query_prefix = "search_query: " if nomic else ""

    async def ingest(self, title: str, text: str) -> dict:
        chunks = chunk_text(text, self.chunk_words, self.overlap_words)
        if not chunks:
            raise ValueError("The document is empty.")
        # Embed everything first, then insert: a failure leaves the DB untouched.
        vectors = await self.llm.embed([self._doc_prefix + c for c in chunks])
        n = len(chunks)
        titles = [f"{title} [{i + 1}/{n}]" if n > 1 else title for i in range(n)]
        ids = self.docs.insert_many(list(zip(titles, chunks, vectors)))
        return {"ids": ids, "chunks": n, "dims": self.docs.dims}

    async def retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        [q] = await self.llm.embed([self._query_prefix + question])
        hits = self.docs.search(q, k, self.max_distance)
        return [RetrievedChunk(rank, d, c) for rank, (d, c) in enumerate(hits, start=1)]

    async def ask(self, question: str, k: int) -> dict:
        contexts = await self.retrieve(question, k)
        answer = await self.llm.generate(build_prompt(question, contexts))
        return {"answer": answer, "model": self.llm.gen_model,
                "contexts": [c.to_dict() for c in contexts], "doc_count": len(self.docs)}

    async def ask_stream(self, question: str, k: int) -> AsyncIterator[dict]:
        """Yields events: one 'contexts', many 'token', then 'done' (or 'error')."""
        try:
            contexts = await self.retrieve(question, k)
            yield {"type": "contexts", "model": self.llm.gen_model,
                   "contexts": [c.to_dict() for c in contexts]}
            async for piece in self.llm.generate_stream(build_prompt(question, contexts)):
                yield {"type": "token", "text": piece}
            yield {"type": "done"}
        except OllamaError as e:
            yield {"type": "error", "message": str(e)}
