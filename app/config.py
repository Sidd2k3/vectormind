"""Settings, read from environment variables with sensible defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8080")))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://127.0.0.1:11434"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    gen_model: str = field(default_factory=lambda: _env("GEN_MODEL", "llama3.2"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    chunk_words: int = field(default_factory=lambda: int(_env("CHUNK_WORDS", "250")))
    overlap_words: int = field(default_factory=lambda: int(_env("OVERLAP_WORDS", "30")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "3")))
    max_distance: float = field(default_factory=lambda: float(_env("MAX_DISTANCE", "0.7")))
    persist: bool = field(default_factory=lambda: _env("PERSIST", "1") != "0")
