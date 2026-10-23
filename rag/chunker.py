"""Split long text into overlapping word windows.

Each chunk gets its own embedding, so retrieval can point at the one
part of a document that answers a question. The overlap means an idea
sitting on a chunk boundary still appears whole in at least one chunk.
"""


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text.strip()]  # short text: keep it whole, line breaks included

    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks
