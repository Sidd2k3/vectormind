# VectorMind

A vector database built from scratch in Python, powering a fully local RAG (Retrieval-Augmented Generation) assistant that answers questions from your own documents.

It implements three nearest-neighbour search algorithms by hand — **Brute Force**, **KD-Tree** and **HNSW** — and lets you compare them side by side. Documents you add are chunked, embedded with a local model through [Ollama](https://ollama.com), indexed in HNSW, and used to answer questions with **Llama 3.2**. Nothing leaves your machine.

## Features

| | |
|---|---|
| **3 search algorithms** | Brute Force (exact baseline), KD-Tree (exact, low-dimensional), HNSW (approximate, scales to high dimensions) |
| **3 distance metrics** | Cosine, Euclidean (L2), Manhattan (L1) |
| **Benchmark** | Median latency of each algorithm plus recall against exact search |
| **Embedding-space view** | 2D PCA projection of the vectors; search hits light up on the plot |
| **HNSW inspector** | Nodes and edges per graph layer |
| **Local RAG** | Chunking → `nomic-embed-text` (768D) → HNSW retrieval → answer from `llama3.2`, streamed token by token, with the source chunks shown |
| **Persistence** | Document chunks and their embeddings survive restarts |
| **REST API** | 14 endpoints with automatic validation and interactive docs at `/docs` |

## How it works

```
Browser (HTML/CSS/JS)  ──HTTP/JSON──►  FastAPI backend  ──HTTP──►  Ollama (localhost:11434)
                                       │                           ├─ nomic-embed-text  (embeddings)
                                       ├─ VectorDB   (16D demo)    └─ llama3.2          (generation)
                                       │    Brute Force · KD-Tree · HNSW
                                       └─ DocumentDB (768D docs)
                                            HNSW · cosine · persisted to disk
```

**Indexing a document**
1. Split the text into 250-word chunks with a 30-word overlap, so ideas on a boundary aren't cut in half.
2. Embed every chunk with `nomic-embed-text` → one 768-dimensional vector per chunk.
3. Store the vectors in the HNSW index (cosine distance).

**Answering a question**
1. Embed the question with the same model.
2. Retrieve the top-k nearest chunks; drop any with cosine distance above 0.7 (nearest is not always relevant).
3. Build a prompt with those chunks. The model bases its answer on them when they're relevant, and otherwise answers from its own general knowledge.
4. Stream the answer from `llama3.2` back to the browser, together with the source chunks.

If no chunk passes the relevance threshold, the model still answers, from its general knowledge, and the UI shows that no document was used.

**On the map.** Each document also gets one marker in the 16D demo space (category `doc`, encoded with the demo keyword encoder) so it appears on the scatter plot, and the documents RAG retrieves are highlighted there. This is visualization only: retrieval always runs on the real 768D document index.

## The search algorithms

| | Brute Force | KD-Tree | HNSW |
|---|---|---|---|
| Result | Exact | Exact | Approximate |
| Query cost | O(N·d) | ~O(log N) in low dimensions, degrades toward O(N) in high dimensions | ~O(log N) |
| Used for | Ground truth, tiny collections | 16D demo vectors | Demo vectors and all documents |

**KD-Tree.** Recursively splits space with median cuts, cycling through the dimensions. A query skips any subtree on the far side of a split when the distance to the splitting plane already exceeds the current k-th best result. Inserts and deletes mark the tree stale, and it is rebuilt balanced on the next query.

**HNSW (Hierarchical Navigable Small World).** A multi-layer proximity graph. Every vector lives on layer 0; each layer above holds roughly 1/M of the layer below. A search greedily descends from the top-layer entry point, then runs a beam search of width `ef` on layer 0.

| Parameter | Value | Meaning |
|---|---|---|
| `M` | 16 | max links per node on upper layers |
| `M0` | 32 | max links per node on layer 0 (2 × M) |
| `ef_construction` | 200 | candidate list size while inserting |
| `ef_search` | 50 | candidate list size while querying (speed vs recall) |

## Design decisions

- **Correct KD-Tree pruning for cosine.** The pruning bound `|query[axis] − node[axis]|` is a valid lower bound for Euclidean and Manhattan distance, but not for cosine distance, which measures angle. Pruning cosine distances directly returned the wrong top-5 on 99 of 100 random test queries. This KD-Tree stores unit-normalized vectors for cosine and searches them with Euclidean distance. For unit vectors `euclidean² = 2 × cosine_distance`, so the ranking is identical and the pruning is exact. A test checks it against brute force.
- **HNSW neighbour-selection heuristic.** A candidate becomes a neighbour only if it's closer to the new node than to any neighbour already chosen. This spreads links in different directions and keeps the graph navigable, instead of linking only within one tight cluster.
- **Soft deletes in HNSW.** Deleted nodes keep routing searches but are never returned. Once they exceed 25% of the graph, it is rebuilt from the live nodes. Removing nodes outright can disconnect parts of the graph.
- **One HNSW graph per metric** in the demo database, so every (algorithm, metric) pair is searched on a graph built with that same metric.
- **Context-first generation.** The prompt tells the model to use the retrieved chunks when they're relevant and its general knowledge otherwise, and to answer naturally without talking about "the context". The chunks used are returned with every answer so they can be checked.
- **Task prefixes for nomic-embed-text.** Chunks are embedded as `search_document: …` and questions as `search_query: …`, the format the model was trained with.
- **Same model, separate store.** Documents are saved per embedding model (`data/documents_<model>.json`). Vectors from different models live in different spaces and are never mixed.
- **Async I/O.** Calls to Ollama are async, so the server keeps serving other requests during a 10–30 second generation. Embeddings are batched (16 per request), and a document is only stored once all of its chunks are embedded.
- **Thread safety.** Each database is guarded by a lock, so a search never reads a half-updated index.

## Setup

**Requirements:** Python 3.10+ and [Ollama](https://ollama.com/download).

```bash
# 1. Pull the models (one time)
ollama pull nomic-embed-text      # embedding model, ~274 MB
ollama pull llama3.2              # language model, ~2 GB

# 2. Install
git clone https://github.com/Sidd2k3/vectormind.git
cd vectormind
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run (Ollama must be running: open the app, or run `ollama serve`)
python -m app
```

Open **http://localhost:8080** for the app and **http://localhost:8080/docs** for the interactive API docs.

The Vector Search tab works without Ollama. The Documents and Ask AI tabs need it.

### Configuration

All settings are optional environment variables.

| Variable | Default | |
|---|---|---|
| `PORT` | `8080` | Server port |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `GEN_MODEL` | `llama3.2` | Generation model (try `llama3.2:1b` on slower laptops) |
| `CHUNK_WORDS` / `OVERLAP_WORDS` | `250` / `30` | Chunking |
| `TOP_K` | `3` | Default chunks retrieved |
| `MAX_DISTANCE` | `0.7` | Relevance threshold (cosine distance) |
| `DATA_DIR` | `./data` | Where documents are saved |
| `PERSIST` | `1` | Set to `0` to keep documents in memory only |

## API

### Demo vectors (16D)

| Method | Endpoint | |
|---|---|---|
| GET | `/search?q=guitar&k=5&metric=cosine&algo=hnsw` | k-NN search by text (`q`) or raw vector (`v=0.1,0.2,…`) |
| POST | `/insert` | Add an item: `{"text": "...", "category": "tech"}` |
| DELETE | `/delete/{id}` | Remove an item |
| GET | `/items` | All items with 2D PCA coordinates |
| GET | `/benchmark?q=guitar&k=5&metric=cosine` | Latency and recall of all three algorithms |
| GET | `/hnsw-info` | HNSW layers, nodes and edges |
| GET | `/stats` | Dimensions, algorithms, metrics |

### Documents and RAG

| Method | Endpoint | |
|---|---|---|
| POST | `/doc/insert` | `{"title": "...", "text": "..."}` → chunk, embed, store |
| GET | `/doc/list` | Stored chunks |
| DELETE | `/doc/delete/{id}` | Remove a chunk |
| POST | `/doc/search` | `{"question": "...", "k": 3}` → retrieval only, no LLM |
| POST | `/doc/ask` | Full RAG answer as JSON |
| POST | `/doc/ask/stream` | Full RAG answer streamed as newline-delimited JSON |
| GET | `/status` | Ollama availability and installed models |

```bash
curl "http://localhost:8080/search?q=rocket+to+mars&k=3&algo=kdtree"

curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes a deadlock?", "k": 3}'
```

## Project structure

```
vectormind/
├── vectordb/               the database, no web or AI dependencies
│   ├── distance.py         cosine, euclidean, manhattan
│   ├── brute_force.py      exact k-NN
│   ├── kdtree.py           KD-Tree k-NN
│   ├── hnsw.py             HNSW graph
│   ├── database.py         VectorDB (demo) and DocumentDB (documents)
│   ├── demo_data.py        16D demo dataset and keyword encoder
│   └── pca.py              2D projection for the plot
├── rag/
│   ├── chunker.py          overlapping word windows
│   ├── ollama_client.py    async Ollama client
│   └── pipeline.py         ingest, retrieve, prompt, generate
├── app/
│   ├── main.py             FastAPI routes
│   ├── schemas.py          request validation
│   ├── config.py           settings
│   └── static/             web UI
└── tests/                  50 tests; Ollama is replaced by a fake
```

## Tests

```bash
pytest
```

The suite checks KD-Tree against brute force for every metric (exact match), HNSW recall@10 ≥ 0.95 on 1,000 random 32D vectors, soft deletion and rebuilds, chunk boundaries and overlap, the RAG flow (with and without relevant context), persistence across restarts, and every API endpoint. Ollama is replaced by a fake, so the tests run without any models installed.

## Demo dataset

The Vector Search tab uses 20 hand-designed 16D vectors. Dimensions 0–3 represent tech, 4–7 space, 8–11 music and 12–15 sports. A few items mix topics on purpose (the ISS is space + tech, chess is sports + tech) so they sit between clusters. Typed queries are mapped to this space by a small keyword encoder that stands in for an embedding model on this dataset only. The Documents and Ask AI tabs use real 768D embeddings.

## Limitations and next steps

- **Hybrid search:** combine vector search with BM25 keyword search for exact terms such as names and error codes.
- **Re-ranking:** retrieve ~20 chunks, then re-score them with a cross-encoder and keep the best few.
- **Smarter chunking:** split on paragraphs and sentences instead of fixed word windows.
- **Strict grounding mode:** an option to answer only from documents (and say "I don't know" otherwise) to rule out hallucinated answers.
- **Evaluation:** recall@k and MRR for retrieval, faithfulness for answers.
- **Scale:** a binary on-disk format instead of JSON, vector quantization to cut memory, sharding across machines.
