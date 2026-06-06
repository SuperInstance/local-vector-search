//! # local-vector-search
//!
//! A local TF-IDF vector search engine for discovering similar repos.
//! No external services required — just point it at a directory and search.

mod document;
mod index;
mod query;
mod search;
mod builder;
mod benchmark;

pub use document::Document;
pub use index::{TfIdfIndex, IndexStats};
pub use query::QueryBuilder;
pub use search::Searcher;
pub use builder::IndexBuilder;
pub use benchmark::Benchmarker;
pub use benchmark::BenchmarkResult;
