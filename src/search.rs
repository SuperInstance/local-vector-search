use crate::document::Document;
use crate::index::TfIdfIndex;

/// Result of a search query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: String,
    pub name: String,
    pub path: String,
    pub score: f64,
}

/// Cosine similarity searcher.
pub struct Searcher<'a> {
    index: &'a TfIdfIndex,
}

impl<'a> Searcher<'a> {
    pub fn new(index: &'a TfIdfIndex) -> Self {
        Self { index }
    }

    /// Search for the top-K documents matching the query.
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        let query_vec = self.index.query_vector(query);
        if query_vec.is_empty() {
            return vec![];
        }
        let query_norm = self.norm(&query_vec);

        let mut results: Vec<SearchResult> = self.index.vectors
            .iter()
            .map(|(doc_id, doc_vec)| {
                let score = self.cosine_similarity(&query_vec, query_norm, doc_vec, doc_id);
                let meta = self.index.doc_meta.get(doc_id).unwrap();
                SearchResult {
                    doc_id: doc_id.clone(),
                    name: meta.name.clone(),
                    path: meta.path.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Search for documents similar to a given document in the index.
    pub fn find_similar(&self, doc_id: &str, k: usize) -> Vec<SearchResult> {
        let doc_vec = match self.index.vectors.get(doc_id) {
            Some(v) => v.clone(),
            None => return vec![],
        };
        let doc_norm = self.index.norms.get(doc_id).copied().unwrap_or(1e-10);

        let mut results: Vec<SearchResult> = self.index.vectors
            .iter()
            .filter(|(id, _)| *id != doc_id)
            .map(|(other_id, other_vec)| {
                let other_norm = self.index.norms.get(other_id).copied().unwrap_or(1e-10);
                let score = self.cosine_similarity_raw(&doc_vec, doc_norm, other_vec, other_norm);
                let meta = self.index.doc_meta.get(other_id).unwrap();
                SearchResult {
                    doc_id: other_id.clone(),
                    name: meta.name.clone(),
                    path: meta.path.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn cosine_similarity(&self, q: &std::collections::HashMap<String, f64>, q_norm: f64, d: &std::collections::HashMap<String, f64>, doc_id: &str) -> f64 {
        let d_norm = self.index.norms.get(doc_id).copied().unwrap_or(1e-10);
        self.cosine_similarity_raw(q, q_norm, d, d_norm)
    }

    fn cosine_similarity_raw(
        &self,
        a: &std::collections::HashMap<String, f64>,
        a_norm: f64,
        b: &std::collections::HashMap<String, f64>,
        b_norm: f64,
    ) -> f64 {
        let dot: f64 = a.iter()
            .filter_map(|(term, wa)| b.get(term).map(|wb| wa * wb))
            .sum();
        dot / (a_norm * b_norm).max(1e-20)
    }

    fn norm(&self, vec: &std::collections::HashMap<String, f64>) -> f64 {
        let sq: f64 = vec.values().map(|v| v * v).sum();
        sq.sqrt().max(1e-10)
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
    fn test_basic_search() {
        let docs = vec![
            make_doc("a", "graph library for rust programming"),
            make_doc("b", "web framework with async http"),
            make_doc("c", "graph algorithm data structures"),
        ];
        let idx = TfIdfIndex::build(&docs);
        let searcher = Searcher::new(&idx);
        let results = searcher.search("graph library", 3);
        assert!(!results.is_empty());
        // "a" or "c" should rank higher than "b"
        assert!(results[0].doc_id == "a" || results[0].doc_id == "c");
    }

    #[test]
    fn test_top_k() {
        let docs = vec![
            make_doc("a", "machine learning deep neural network"),
            make_doc("b", "machine learning classification"),
            make_doc("c", "web server http"),
        ];
        let idx = TfIdfIndex::build(&docs);
        let searcher = Searcher::new(&idx);
        let results = searcher.search("machine learning", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_empty_query() {
        let docs = vec![make_doc("a", "some text")];
        let idx = TfIdfIndex::build(&docs);
        let searcher = Searcher::new(&idx);
        let results = searcher.search("", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_similar() {
        let docs = vec![
            make_doc("a", "graph library rust"),
            make_doc("b", "graph algorithms data structures"),
            make_doc("c", "web framework async"),
        ];
        let idx = TfIdfIndex::build(&docs);
        let searcher = Searcher::new(&idx);
        let similar = searcher.find_similar("a", 2);
        assert!(!similar.is_empty());
        // "b" should be more similar to "a" than "c"
        assert!(similar[0].doc_id == "b");
    }
}
