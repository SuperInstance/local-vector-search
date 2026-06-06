use crate::builder::IndexBuilder;
use crate::index::TfIdfIndex;
use crate::search::{SearchResult, Searcher};
use std::time::Instant;

/// Benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub num_docs: u32,
    pub index_build_ms: u64,
    pub index_size_bytes: usize,
    pub index_size_mb: f64,
    pub avg_query_us: u64,
    pub p50_query_us: u64,
    pub p99_query_us: u64,
    pub max_query_us: u64,
    pub memory_rss_kb: u64,
    pub sample_results: Vec<(String, Vec<(String, f64)>)>,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Benchmark Results ===")?;
        writeln!(f, "Documents:      {}", self.num_docs)?;
        writeln!(f, "Index build:    {} ms", self.index_build_ms)?;
        writeln!(f, "Index size:     {:.2} MB ({} bytes)", self.index_size_mb, self.index_size_bytes)?;
        writeln!(f, "Avg query:      {} µs ({:.2} ms)", self.avg_query_us, self.avg_query_us as f64 / 1000.0)?;
        writeln!(f, "P50 query:      {} µs ({:.2} ms)", self.p50_query_us, self.p50_query_us as f64 / 1000.0)?;
        writeln!(f, "P99 query:      {} µs ({:.2} ms)", self.p99_query_us, self.p99_query_us as f64 / 1000.0)?;
        writeln!(f, "Max query:      {} µs ({:.2} ms)", self.max_query_us, self.max_query_us as f64 / 1000.0)?;
        writeln!(f, "Memory (RSS):   {} KB ({:.1} MB)", self.memory_rss_kb, self.memory_rss_kb as f64 / 1024.0)?;
        Ok(())
    }
}

/// Runs benchmarks against the index.
pub struct Benchmarker;

impl Benchmarker {
    /// Run a full benchmark: build index, run queries, measure everything.
    pub fn run(root: &str, queries: &[&str], top_k: usize) -> BenchmarkResult {
        // Build index
        let start = Instant::now();
        let builder = IndexBuilder::new(root);
        let index = builder.build();
        let build_ms = start.elapsed().as_millis() as u64;

        let stats = index.stats();
        let index_bytes = stats.index_bytes;

        // Run queries
        let searcher = Searcher::new(&index);
        let mut query_times: Vec<u64> = Vec::with_capacity(queries.len());
        let mut sample_results: Vec<(String, Vec<(String, f64)>)> = Vec::new();

        for query in queries {
            let start = Instant::now();
            let results = searcher.search(query, top_k);
            let elapsed = start.elapsed().as_micros() as u64;
            query_times.push(elapsed);

            let top: Vec<(String, f64)> = results.iter()
                .take(3)
                .map(|r| (r.name.clone(), r.score))
                .collect();
            sample_results.push((query.to_string(), top));
        }

        query_times.sort();
        let avg_query_us = query_times.iter().sum::<u64>() / query_times.len().max(1) as u64;
        let p50_idx = query_times.len() / 2;
        let p99_idx = (query_times.len() as f64 * 0.99) as usize;
        let p50_query_us = query_times.get(p50_idx).copied().unwrap_or(0);
        let p99_query_us = query_times.get(p99_idx.min(query_times.len() - 1)).copied().unwrap_or(0);
        let max_query_us = *query_times.last().unwrap_or(&0);

        // Estimate memory from index size (rough)
        let memory_kb = index_bytes / 1024;

        BenchmarkResult {
            num_docs: stats.num_docs,
            index_build_ms: build_ms,
            index_size_bytes: index_bytes,
            index_size_mb: index_bytes as f64 / 1_048_576.0,
            avg_query_us,
            p50_query_us,
            p99_query_us,
            max_query_us,
            memory_rss_kb: memory_kb as u64,
            sample_results,
        }
    }
}
