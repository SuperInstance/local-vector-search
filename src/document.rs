use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a searchable document (typically a repo).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier (repo name or path).
    pub id: String,
    /// Display name.
    pub name: String,
    /// Full path on disk.
    pub path: String,
    /// Tokenized text features ready for TF-IDF.
    pub tokens: Vec<String>,
    /// Original raw text fields for reference.
    pub raw_text: HashMap<String, String>,
}

impl Document {
    /// Create a new document with the given id, name, path, and pre-tokenized features.
    pub fn new(id: impl Into<String>, name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            path: path.into(),
            tokens: Vec::new(),
            raw_text: HashMap::new(),
        }
    }

    /// Add a text field; it will be tokenized and appended to the token list.
    pub fn add_field(&mut self, field_name: &str, text: &str) {
        self.raw_text.insert(field_name.to_string(), text.to_string());
        let tokens = tokenize(text);
        self.tokens.extend(tokens);
    }

    /// Get the term frequency map for this document.
    pub fn term_frequencies(&self) -> HashMap<String, f64> {
        let mut tf = HashMap::new();
        let total = self.tokens.len() as f64;
        if total == 0.0 {
            return tf;
        }
        for token in &self.tokens {
            *tf.entry(token.clone()).or_insert(0.0) += 1.0;
        }
        for v in tf.values_mut() {
            *v /= total;
        }
        tf
    }
}

/// Simple tokenizer: lowercase, split on non-alphanumeric, stem English words.
pub fn tokenize(text: &str) -> Vec<String> {
    use rust_stemmers::{Algorithm, Stemmer};
    let stemmer = Stemmer::create(Algorithm::English);
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() > 1)
        .map(|s| stemmer.stem(s).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_new() {
        let doc = Document::new("test-repo", "Test Repo", "/path/to/repo");
        assert_eq!(doc.id, "test-repo");
        assert_eq!(doc.name, "Test Repo");
        assert!(doc.tokens.is_empty());
    }

    #[test]
    fn test_add_field() {
        let mut doc = Document::new("r1", "R1", "/tmp/r1");
        doc.add_field("readme", "A graph library for Rust");
        assert!(!doc.tokens.is_empty());
        assert!(doc.raw_text.contains_key("readme"));
    }

    #[test]
    fn test_term_frequencies() {
        let mut doc = Document::new("r1", "R1", "/tmp/r1");
        doc.add_field("desc", "graph graph graph library");
        let tf = doc.term_frequencies();
        assert!(!tf.is_empty());
        // "graph" should have highest frequency
        let graph_freq = tf.keys().find(|k| k.contains("graph")).unwrap();
        assert!(tf[graph_freq] > 0.0);
    }

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_special_chars() {
        let tokens = tokenize("react-typescript next.js vue.js");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_document_tokens_dedup_not_needed() {
        let mut doc = Document::new("r1", "R1", "/tmp/r1");
        doc.add_field("a", "test test test");
        doc.add_field("b", "test test");
        // tokens can repeat, term_frequencies handles it
        assert!(doc.tokens.len() >= 5);
    }
}
