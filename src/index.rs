use crate::document::{Document, tokenize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// TF-IDF index over a collection of documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfIdfIndex {
    /// Maps term -> document frequency (number of docs containing term).
    pub df: HashMap<String, u32>,
    /// Maps doc_id -> TF-IDF vector (term -> weight).
    pub vectors: HashMap<String, HashMap<String, f64>>,
    /// Number of documents.
    pub num_docs: u32,
    /// Document metadata (id, name, path).
    pub doc_meta: HashMap<String, DocMeta>,
    /// Norms for cosine similarity: doc_id -> L2 norm.
    pub norms: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    pub id: String,
    pub name: String,
    pub path: String,
}

/// Statistics about the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub num_docs: u32,
    pub vocab_size: usize,
    pub avg_tokens_per_doc: f64,
    pub index_bytes: usize,
}

impl TfIdfIndex {
    /// Build a TF-IDF index from a list of documents.
    pub fn build(documents: &[Document]) -> Self {
        let num_docs = documents.len() as u32;
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut tf_maps: HashMap<String, HashMap<String, f64>> = HashMap::new();
        let mut doc_meta: HashMap<String, DocMeta> = HashMap::new();
        let mut total_tokens = 0usize;

        // Compute term frequencies
        for doc in documents {
            total_tokens += doc.tokens.len();
            let tf = doc.term_frequencies();
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            tf_maps.insert(doc.id.clone(), tf);
            doc_meta.insert(
                doc.id.clone(),
                DocMeta {
                    id: doc.id.clone(),
                    name: doc.name.clone(),
                    path: doc.path.clone(),
                },
            );
        }

        // Compute TF-IDF vectors
        let mut vectors: HashMap<String, HashMap<String, f64>> = HashMap::new();
        let mut norms: HashMap<String, f64> = HashMap::new();

        for (doc_id, tf) in &tf_maps {
            let mut vec = HashMap::new();
            let mut norm_sq = 0.0f64;
            for (term, &freq) in tf {
                let idf = ((num_docs as f64 + 1.0) / (*df.get(term).unwrap_or(&1) as f64 + 1.0)).ln() + 1.0;
                let weight = freq * idf;
                vec.insert(term.clone(), weight);
                norm_sq += weight * weight;
            }
            norms.insert(doc_id.clone(), norm_sq.sqrt().max(1e-10));
            vectors.insert(doc_id.clone(), vec);
        }

        Self {
            df,
            vectors,
            num_docs,
            doc_meta,
            norms,
        }
    }

    /// Serialize the index to binary bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("Serialization error: {}", e))
    }

    /// Deserialize an index from binary bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| format!("Deserialization error: {}", e))
    }

    /// Get statistics about this index.
    pub fn stats(&self) -> IndexStats {
        let vocab_size = self.df.len();
        let avg = if self.num_docs > 0 {
            self.vectors.values().map(|v| v.len()).sum::<usize>() as f64 / self.num_docs as f64
        } else {
            0.0
        };
        IndexStats {
            num_docs: self.num_docs,
            vocab_size,
            avg_tokens_per_doc: avg,
            index_bytes: self.to_bytes().unwrap_or_default().len(),
        }
    }

    /// Compute a TF-IDF vector for a query string.
    pub fn query_vector(&self, query: &str) -> HashMap<String, f64> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return HashMap::new();
        }
        let total = tokens.len() as f64;
        let mut tf: HashMap<String, f64> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }
        for v in tf.values_mut() {
            *v /= total;
        }
        let mut vec = HashMap::new();
        for (term, freq) in &tf {
            let idf = ((self.num_docs as f64 + 1.0) / (*self.df.get(term).unwrap_or(&1) as f64 + 1.0)).ln() + 1.0;
            vec.insert(term.clone(), freq * idf);
        }
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, text: &str) -> Document {
        let mut doc = Document::new(id, id, format!("/tmp/{}", id));
        doc.add_field("text", text);
        doc
    }

    #[test]
    fn test_build_index() {
        let docs = vec![
            make_doc("a", "graph library for rust"),
            make_doc("b", "web framework with async"),
            make_doc("c", "graph algorithm visualization"),
        ];
        let idx = TfIdfIndex::build(&docs);
        assert_eq!(idx.num_docs, 3);
        assert!(!idx.vectors.is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let docs = vec![
            make_doc("a", "graph library"),
            make_doc("b", "web framework"),
        ];
        let idx = TfIdfIndex::build(&docs);
        let bytes = idx.to_bytes().unwrap();
        let idx2: TfIdfIndex = TfIdfIndex::from_bytes(&bytes).unwrap();
        assert_eq!(idx2.num_docs, 2);
        assert_eq!(idx2.df.len(), idx.df.len());
    }

    #[test]
    fn test_query_vector() {
        let docs = vec![make_doc("a", "graph library"), make_doc("b", "web framework")];
        let idx = TfIdfIndex::build(&docs);
        let qv = idx.query_vector("find graph libraries");
        assert!(!qv.is_empty());
    }

    #[test]
    fn test_stats() {
        let docs = vec![make_doc("a", "test document one"), make_doc("b", "test document two")];
        let idx = TfIdfIndex::build(&docs);
        let stats = idx.stats();
        assert_eq!(stats.num_docs, 2);
        assert!(stats.vocab_size > 0);
        assert!(stats.index_bytes > 0);
    }

    #[test]
    fn test_empty_documents() {
        let docs: Vec<Document> = vec![];
        let idx = TfIdfIndex::build(&docs);
        assert_eq!(idx.num_docs, 0);
        assert!(idx.vectors.is_empty());
    }
}
