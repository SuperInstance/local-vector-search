# local-vector-search

A local TF-IDF vector search engine for repo discovery. No external services, no databases, no API keys — just point it at a directory and search.

## Results (609 repos, real benchmark)

```
Documents:      609
Index build:    83 ms
Index size:     2.07 MB (2,165,762 bytes)
Avg query:      158 µs (0.16 ms)
P50 query:      144 µs (0.14 ms)
P99 query:      302 µs (0.30 ms)
Max query:      302 µs (0.30 ms)
```

### Targets vs Actuals

| Metric | Target | Actual | ✅/❌ |
|--------|--------|--------|-------|
| Query latency | <100ms | **0.16ms avg** | ✅ (600x faster) |
| Index size | <50MB | **2.07MB** | ✅ (24x smaller) |
| Index build | — | **83ms** | ✅ |
| Repos indexed | 600+ | **609** | ✅ |

### Sample Queries

```
Q: "find graph libraries with algorithms"
  → graph-walker-go (0.216), topological-sort-agent-rs (0.204), ternary-search (0.203)

Q: "database sql query storage"
  → ternary-database (0.322), torch-vector-search (0.131), pincher (0.080)

Q: "crypto blockchain cryptography"
  → ternary-blockchain (0.189), ternary-cipher (0.072), superinstance-vectorize (0.037)

Q: "game engine physics rendering"
  → ternary-game-theory (0.294), ternary-games (0.226), ternary-visualizer (0.188)
```

## Architecture

```
Document    → represents a repo with text features (name, readme, deps, extensions)
Index       → builds a TF-IDF index over all documents
Searcher    → cosine similarity search returning top-K results
QueryBuilder→ natural language queries ("find me graph libraries with tests")
IndexBuilder→ scans a directory, builds the index, serializes to binary
Benchmarker → measures query latency, index build time, memory usage
```

### Feature Extraction

For each repo, the `IndexBuilder` extracts:
- **Name** — the directory name itself
- **README** — first 2000 chars of README.md/README.txt
- **Dependencies** — parsed from Cargo.toml, package.json, requirements.txt
- **File extensions** — all unique extensions found in the repo
- **Key files** — presence of main.rs, lib.rs, tests/, examples/, etc.

All features are tokenized, lowercased, and stemmed (English Snowball stemmer) before indexing.

### TF-IDF + Cosine Similarity

1. Term frequency (TF) normalized per document
2. Inverse document frequency (IDF) with smoothing: `idf = ln((N+1)/(df+1)) + 1`
3. Documents and queries represented as TF-IDF vectors
4. Ranked by cosine similarity

## Usage

### CLI

```bash
# Build and run against ~/repos
cargo run --release -- ~/repos

# Or use the binary directly
./target/release/lvs ~/repos
```

### Library

```rust
use local_vector_search::{IndexBuilder, Searcher, QueryBuilder};

// Build index from directory
let builder = IndexBuilder::new("/path/to/repos");
let index = builder.build();

// Search
let searcher = Searcher::new(&index);
let results = searcher.search("graph algorithms", 5);
for r in &results {
    println!("{:.4}  {} ({})", r.score, r.name, r.path);
}

// Natural language query
let qb = QueryBuilder::parse("find me graph libraries with tests");
let query = qb.build();
let results = searcher.search(&query, 10);

// Find similar repos
let similar = searcher.find_similar("my-graph-lib", 5);

// Serialize / deserialize
let bytes = index.to_bytes().unwrap();
let index2 = TfIdfIndex::from_bytes(&bytes).unwrap();
```

## Performance

- **Index build**: ~83ms for 609 repos (parallel feature extraction with rayon)
- **Query**: ~160µs average (pure in-memory cosine similarity, no disk I/O)
- **Memory**: ~2MB for the full index (bincode serialization)
- **Serialization**: Full round-trip in microseconds via bincode

The index is small enough to fit in L3 cache on modern hardware, which is why queries are sub-millisecond.

## Dependencies

- `serde` + `bincode` — serialization
- `rayon` — parallel repo scanning
- `walkdir` — directory traversal
- `rust-stemmers` — English Snowball stemming

## Tests

23 tests covering:
- Document creation, tokenization, term frequencies
- Index building, serialization roundtrip, query vectors
- Search: basic search, top-K, empty queries, find-similar
- Query builder: natural language parsing, exclusions, chaining
- Builder: initialization, nonexistent directory handling

```bash
cargo test
```
