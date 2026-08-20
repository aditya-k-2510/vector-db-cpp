# VectorDB — Vector Database from Scratch in C++

A fully working vector database built from scratch in C++, with no external dependencies beyond a single-header HTTP library. Implements three search algorithms side-by-side, a full REST API, a live web UI with PCA visualization, and a RAG pipeline powered by a local LLM via Ollama.

> Built to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

---

## Features

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **Correct Cosine KD-Tree** | Pruning disabled for cosine queries — exact results, no silent wrong answers |
| **HNSW Algorithm 4** | Diversity neighbor selection heuristic from Malkov & Yashunin — not naive top-M |
| **Reader-Writer Concurrency** | `std::shared_mutex` — searches run concurrently, writes are exclusive |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of vector space — watch semantic clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Question → HNSW retrieval → local LLM → answer |
| **Scale Benchmark** | `./db --bench` — latency + recall@k across N=1k→50k, with ef sweep |
| **Full REST API** | CRUD endpoints for vectors and documents |

---

## Project Structure

```
VectorDB/
├── main.cpp        ← entire backend: algorithms, HTTP server, RAG pipeline (~1260 lines)
├── httplib.h       ← single-header HTTP server library (cpp-httplib, bundled)
├── index.html      ← frontend: PCA scatter plot, document tab, AI chat tab
└── README.md       ← this file
```

No build system, no package manager, no dependencies to install. One file, one binary.

---

## How to Run

### Prerequisites

**A C++17 compiler.** That's the only hard requirement.

- **macOS:** `xcode-select --install` (installs clang++)
- **Linux:** `sudo apt install g++`
- **Windows:** Install MSYS2 from https://www.msys2.org, then `pacman -S mingw-w64-ucrt-x86_64-gcc`. Add `C:\msys64\ucrt64\bin` to your PATH.

**Ollama** (optional — needed only for document embedding and RAG, not for the demo vector search):

Download from https://ollama.com, then:
```bash
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB   — generation model
```

### Compile

**macOS / Linux:**
```bash
g++ -std=c++17 -O2 main.cpp -o db
```

**Windows:**
```powershell
g++ -std=c++17 -O2 main.cpp -o db -lws2_32
```

Takes about 10–20 seconds. Produces a single `db` binary.

> **VS Code showing red squiggles on `shared_mutex`?** Add `.vscode/c_cpp_properties.json`:
> ```json
> { "configurations": [{ "name": "Mac", "cppStandard": "c++17" }], "version": 4 }
> ```
> The code compiles fine regardless — this just fixes IntelliSense.

### Run

```bash
./db
```

Output:
```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open `http://localhost:8080`. Make sure `index.html` is in the same directory as the binary.

---

## Using the Application

### Tab 1 — Search (Demo Vectors)

20 pre-loaded 16-dimensional vectors across 4 semantic categories. The dimensions are laid out intentionally: dims 0–3 encode CS/Algorithms, 4–7 Math, 8–11 Food, 12–15 Sports — so category similarity maps directly to geometric proximity.

- Type any concept: `binary tree`, `sushi`, `basketball`, `calculus`
- Pick algorithm (HNSW / KD-Tree / Brute Force) and metric (Cosine / Euclidean / Manhattan)
- **⚡ SEARCH** returns results with distances; the matching point glows on the PCA scatter plot
- **▶ COMPARE ALL ALGOS** runs all three and shows their latencies side by side

### Tab 2 — Documents (Real Embeddings)

1. Enter a title and paste any text
2. Click **⚡ EMBED & INSERT**

Long documents are split into overlapping 250-word chunks (30-word overlap). Each chunk gets its own 768D embedding from Ollama and is stored in a separate HNSW index. When fewer than 10 chunks are stored, brute force is used instead of HNSW (graph traversal has overhead that isn't worth it at tiny scale).

### Tab 3 — Ask AI (RAG Pipeline)

1. Insert documents in Tab 2 first
2. Type a question, click **🤖 ASK AI**

What happens:
```
1. Question  →  embedded with nomic-embed-text (768D vector)
2. HNSW      →  finds top-k semantically similar chunks (cosine distance < 0.7)
3. Chunks    →  sent as context to llama3.2
4. llama3.2  →  generates an answer grounded in your documents
```

The answer streams in with a typewriter effect. Context chips show which chunks were retrieved.

---

## Architecture

### Class Layout

```
BruteForce      O(N·d)        Exact. Scans every vector. Ground truth for benchmarking.

KDTree          O(log N)      Exact. Axis-aligned space partitioning.
                              Prunes far subtrees for Euclidean/Manhattan.
                              For cosine: visits both subtrees (no valid per-axis
                              bound exists for angular distance) — still exact, O(N).
                              Degrades toward O(N) at high dimensions.

HNSW            ~O(log N)     Approximate. Multilayer navigable small-world graph.
                              Recall tunable via ef parameter.

VectorDB                      Unified interface over all 3 indexes (16D demo vectors).
DocumentDB                    HNSW index for Ollama embeddings (768D).
OllamaClient                  HTTP client → /api/embeddings + /api/generate
```

### Concurrency Model

`cpp-httplib` serves each request on a thread-pool thread. Both `VectorDB` and `DocumentDB` use `std::shared_mutex`:

- **Read paths** (`search`, `benchmark`, `all`, `hnswInfo`, `size`) → `std::shared_lock` — any number run simultaneously
- **Write paths** (`insert`, `remove`) → `std::unique_lock` — exclusive, waits for readers to drain

Locking is at the database boundary, not inside individual indexes. This keeps all three indexes mutually consistent — a search never sees a vector present in the store but not yet in the HNSW graph.

A plain `std::mutex` would be correct but would fully serialize all requests. For a read-heavy workload like a vector database, `shared_mutex` turns lock contention from "every request queues" into "only writes block."

---

## HNSW Implementation

The same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.

### Structure

Nodes are inserted into a multilayer graph. Each node gets assigned a random maximum layer drawn from `floor(-log(uniform) * mL)` where `mL = 1/log(M)` — this gives an exponential distribution so higher layers are exponentially sparser. Layer 0 has all nodes and M₀ = 2M connections; upper layers have M connections.

### Insert

1. Start at the top layer, greedily descend to the nearest node
2. Drop a layer, repeat until reaching the node's assigned max layer
3. From max layer down to 0: beam search with `ef_construction=200`, then connect to neighbors using the diversity heuristic

### Search

Greedy descent from the top layer down to layer 1 (tracking the single nearest node per layer), then beam search at layer 0 with width `ef`.

### Neighbor Selection — Algorithm 4 (Malkov & Yashunin)

Naive top-M selection picks the M closest candidates. In clustered data, they all sit in the same direction — the graph loses long-range shortcut edges and gets trapped in local neighborhoods.

The diversity heuristic keeps a candidate only if it is **closer to the base point than to every already-selected neighbor**. This spreads edges across directions and preserves graph navigability. Skipped candidates backfill under-connected nodes (`keepPrunedConnections`). The same heuristic is applied when a neighbor's connection list overflows during insert.

### ef Tradeoff

`ef` is the beam width at search time. Larger ef = more candidates examined = higher recall, higher latency. This is the fundamental approximate nearest-neighbor tradeoff — every production vector database exposes it (Pinecone's `top_k` tuning, pgvector's `hnsw.ef_search`, Weaviate's `ef`).

---

## KD-Tree and Cosine Distance

The KD-tree pruning test — `|q[axis] − node[axis]| < current_kth_best` — is a valid lower bound on the true distance only for Euclidean (per-axis diff ≤ L2) and Manhattan (per-axis diff ≤ L1).

Cosine distance is angular and has no per-axis geometric bound. Applying this pruning to cosine queries silently drops true nearest neighbors and returns wrong results. This implementation disables pruning for cosine (`canPrune = metric != "cosine"`) and visits both subtrees instead — exact, but O(N). Results are byte-identical to brute force on all cosine queries.

The production approach: normalize all vectors to unit length and index with L2. For unit vectors, `cosine_dist = ‖a−b‖²/2`, so L2 order equals cosine order and pruning becomes valid again.

---

## Scale Benchmark

The web UI benchmark runs on 20 vectors — brute force wins there because graph traversal overhead dominates at tiny scale. The CLI benchmark shows what happens at realistic N.

```bash
./db --bench                # default suite
./db --bench 50000 16       # custom: N=50000, 16 dimensions
./db --bench 100000 128     # custom: N=100000, 128 dimensions
```

Data is generated as clustered Gaussians (32 clusters) — more realistic than uniform noise, because real embeddings cluster by semantics. Queries are drawn from the same distribution as the data; querying from a different distribution makes recall numbers pessimistic.

### Results (single-core, g++ -O2, k=10, 50 queries)

| N | dims | BruteForce | KD-Tree | HNSW ef=50 | HNSW speedup | recall@10 |
|---|---|---|---|---|---|---|
| 1,000 | 16 | 56 µs | 19 µs | 41 µs | ~1× | 1.000 |
| 10,000 | 16 | 869 µs | 101 µs | 84 µs | 10× | 1.000 |
| 50,000 | 16 | 4,912 µs | 495 µs | 225 µs | **22×** | 1.000 |
| 20,000 | 128 | 4,187 µs | **7,133 µs** | 347 µs | 12× | 0.996 |

**The crossover is real.** At N=1k, HNSW's graph overhead buys nothing. By N=50k it's 22× faster than exact search at perfect recall.

**The curse of dimensionality, demonstrated.** At 128D, KD-Tree is slower than brute force (7.1ms vs 4.2ms). Pruning stops working because in high dimensions the axis-aligned bound almost never beats the current best — the tree degenerates into a full traversal plus pointer-chasing overhead. HNSW's graph-based approach doesn't rely on geometric pruning and keeps its advantage. This is why production vector databases (768–1536 dims) use HNSW or IVF, never KD-trees.

### Recall@k vs ef (N=50k, 16D)

| ef | latency | recall@10 |
|---|---|---|
| 10 | 87 µs | 0.960 |
| 20 | 122 µs | 0.992 |
| 50 | 225 µs | 1.000 |
| 200 | 522 µs | 1.000 |

HNSW is approximate — measuring recall is non-negotiable. An index that returns garbage can be made arbitrarily fast. The recall@10 metric (fraction of true k nearest neighbors returned, with brute force as ground truth) makes the speed/accuracy tradeoff concrete.

---

## REST API

Base URL: `http://localhost:8080`

### Demo Vector Endpoints

| Method | Endpoint | Params / Body | Description |
|---|---|---|---|
| `GET` | `/search` | `v`, `k`, `metric`, `algo` | K-NN search |
| `POST` | `/insert` | `{"metadata","category","embedding"}` | Insert vector |
| `DELETE` | `/delete/:id` | — | Delete by ID |
| `GET` | `/items` | — | List all vectors |
| `GET` | `/benchmark` | `v`, `k`, `metric` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | — | Graph structure, layers, edges |
| `GET` | `/stats` | — | Count, dims, available algos/metrics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title","text"}` | Chunk, embed, and store |
| `GET` | `/doc/list` | — | List chunks (id, title, 120-char preview, word count) |
| `DELETE` | `/doc/delete/:id` | — | Delete chunk |
| `POST` | `/doc/search` | `{"question","k"}` | Retrieve similar chunks (no LLM) |
| `POST` | `/doc/ask` | `{"question","k"}` | Full RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status, model names, doc/demo counts |

### Examples

```bash
# Search
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

# Ask a question
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Known Limitations

- **No persistence** — indexes are in-memory and rebuilt on restart. A production fix would serialize the HNSW graph to disk on insert and reload on startup.
- **HNSW remove is O(N·E)** — scans all nodes to strip incoming edges. Production systems use tombstones and periodic rebuilds.
- **KD-Tree insertion order** — no median splitting, can degenerate on sorted input. `rebuild()` mitigates after deletes.
- **No SIMD** — distance loops are scalar. A production system vectorizes with AVX2/NEON, which is where most real-world speed comes from at high dimensions.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` |
| VS Code squiggles on `shared_mutex` | Add `.vscode/c_cpp_properties.json` with `"cppStandard": "c++17"` |
| Port 8080 in use (macOS) | System Settings → AirDrop & Handoff → AirPlay Receiver → off |
| Port 8080 in use (Linux) | `lsof -i :8080` then `kill -9 <PID>` |
| LLM is slow | Switch to `llama3.2:1b` — pull it with `ollama pull llama3.2:1b` and update `genModel` in `main.cpp` |

---

## License

MIT