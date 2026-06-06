/// Natural language query builder.
pub struct QueryBuilder {
    keywords: Vec<String>,
    exclude: Vec<String>,
    boost_fields: Vec<String>,
}

impl QueryBuilder {
    pub fn new() -> Self {
        Self {
            keywords: Vec::new(),
            exclude: Vec::new(),
            boost_fields: Vec::new(),
        }
    }

    /// Parse a natural language query into structured form.
    pub fn parse(natural: &str) -> Self {
        let mut qb = Self::new();
        let lower = natural.to_lowercase();

        // Remove stop phrases
        let cleaned = lower
            .replace("find me", "")
            .replace("find", "")
            .replace("show me", "")
            .replace("show", "")
            .replace("search for", "")
            .replace("search", "")
            .replace("looking for", "")
            .replace("i need", "")
            .replace("i want", "")
            .replace("repos", "")
            .replace("repositories", "")
            .replace("libraries", "")
            .replace("library", "")
            .replace("that", "")
            .replace(" and ", " ")
            .replace("please", "")
            .replace(" with ", " ");

        // Handle "without X" / "no X" exclusions
        let parts: Vec<&str> = cleaned.split_whitespace().filter(|s| !s.is_empty()).collect();
        let mut skip_next = false;
        for (i, part) in parts.iter().enumerate() {
            if skip_next {
                qb.exclude.push((*part).to_string());
                skip_next = false;
                continue;
            }
            if *part == "without" || *part == "no" || *part == "not" || *part == "excluding" {
                skip_next = true;
                continue;
            }
            qb.keywords.push((*part).to_string());
        }

        qb
    }

    /// Add a keyword to the query.
    pub fn keyword(mut self, kw: &str) -> Self {
        self.keywords.push(kw.to_string());
        self
    }

    /// Exclude a term from results.
    pub fn exclude(mut self, term: &str) -> Self {
        self.exclude.push(term.to_string());
        self
    }

    /// Build the final query string for the searcher.
    pub fn build(&self) -> String {
        let mut parts = self.keywords.clone();
        // Add exclusions as negative terms (they'll just not boost, since TF-IDF is positive)
        // We include exclusions by removing them from the query rather than negative matching
        parts.join(" ")
    }

    /// Get the keywords.
    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }

    /// Get the excluded terms.
    pub fn excluded(&self) -> &[String] {
        &self.exclude
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let qb = QueryBuilder::parse("find me graph libraries");
        let kws = qb.keywords();
        assert!(kws.contains(&"graph".to_string()));
    }

    #[test]
    fn test_parse_with_exclusion() {
        let qb = QueryBuilder::parse("find graph libraries without charts");
        assert!(qb.keywords().contains(&"graph".to_string()));
        assert!(qb.excluded().contains(&"charts".to_string()));
    }

    #[test]
    fn test_parse_natural_language() {
        let qb = QueryBuilder::parse("I want repos that have async http server and tests");
        let built = qb.build();
        assert!(built.contains("async"));
        assert!(built.contains("http"));
        assert!(built.contains("server"));
        assert!(built.contains("tests"));
    }

    #[test]
    fn test_builder_chain() {
        let qb = QueryBuilder::new().keyword("rust").keyword("graph").exclude("python");
        assert_eq!(qb.keywords().len(), 2);
        assert_eq!(qb.excluded().len(), 1);
    }

    #[test]
    fn test_build_string() {
        let qb = QueryBuilder::parse("show me graph libraries with tests");
        let s = qb.build();
        assert!(!s.is_empty());
    }
}
