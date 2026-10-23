"""A stand-in for OllamaClient so tests run without Ollama installed.

Embeddings are bag-of-words vectors over a small vocabulary, so texts
sharing words are close in cosine distance, which is enough to test retrieval.
"""

import numpy as np

VOCAB = ["deadlock", "process", "processes", "wait", "waiting", "forever", "bread",
         "bake", "degrees", "minutes", "oven", "music", "guitar"]


class FakeOllama:
    def __init__(self, embed_model="nomic-embed-text", gen_model="fake-llm", available=True):
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.available = available
        self.embedded: list[str] = []
        self.last_prompt = None

    def _vec(self, text: str) -> np.ndarray:
        words = text.lower().replace(":", " ").replace(".", " ").split()
        v = np.array([words.count(w) for w in VOCAB], dtype=np.float32)
        v = np.append(v, 0.01)  # avoid an all-zero vector
        return v

    async def is_available(self):
        return self.available

    async def list_models(self):
        return [self.embed_model, self.gen_model] if self.available else []

    async def embed(self, texts):
        self.embedded.extend(texts)
        return [self._vec(t) for t in texts]

    async def generate(self, prompt):
        self.last_prompt = prompt
        return "ANSWER: grounded [1]"

    async def generate_stream(self, prompt):
        self.last_prompt = prompt
        for piece in ["ANSWER:", " grounded", " [1]"]:
            yield piece

    async def aclose(self):
        pass
