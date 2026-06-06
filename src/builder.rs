use crate::document::Document;
use crate::index::TfIdfIndex;
use rayon::prelude::*;
use std::path::Path;
use walkdir::WalkDir;

/// Builds an index by scanning a directory of repos.
pub struct IndexBuilder {
    root: String,
    max_depth: usize,
    max_repos: Option<usize>,
}

impl IndexBuilder {
    pub fn new(root: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            max_depth: 3,
            max_repos: None,
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn max_repos(mut self, n: usize) -> Self {
        self.max_repos = Some(n);
        self
    }

    /// Scan the directory and collect documents.
    pub fn scan(&self) -> Vec<Document> {
        let root = Path::new(&self.root);
        if !root.exists() {
            eprintln!("Warning: {} does not exist", self.root);
            return vec![];
        }

        // Collect repo paths (directories containing .git or just immediate subdirs)
        let mut repo_dirs: Vec<std::path::PathBuf> = Vec::new();
        for entry in std::fs::read_dir(root).unwrap_or_else(|e| {
            eprintln!("Error reading dir: {}", e);
            std::fs::read_dir(".").unwrap()
        }).flatten() {
            if entry.path().is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.') {
                    continue;
                }
                repo_dirs.push(entry.path());
            }
        }

        if let Some(max) = self.max_repos {
            repo_dirs.truncate(max);
        }

        // Extract features from each repo in parallel
        let documents: Vec<Document> = repo_dirs
            .par_iter()
            .filter_map(|dir| {
                let name = dir.file_name()?.to_string_lossy().to_string();
                let path = dir.to_string_lossy().to_string();
                let mut doc = Document::new(&name, &name, &path);

                // Extract features from various sources
                let readme = self.extract_readme(dir);
                if let Some(text) = &readme {
                    doc.add_field("readme", text);
                }

                let deps = self.extract_dependencies(dir);
                if !deps.is_empty() {
                    doc.add_field("deps", &deps.join(" "));
                }

                let extensions = self.extract_extensions(dir);
                if !extensions.is_empty() {
                    doc.add_field("extensions", &extensions.join(" "));
                }

                let filenames = self.extract_key_filenames(dir);
                if !filenames.is_empty() {
                    doc.add_field("files", &filenames.join(" "));
                }

                // Always add the name itself as a feature
                doc.add_field("name", &name);

                Some(doc)
            })
            .collect();

        documents
    }

    /// Build the index from scanned documents.
    pub fn build(&self) -> TfIdfIndex {
        let docs = self.scan();
        println!("Scanned {} repos", docs.len());
        TfIdfIndex::build(&docs)
    }

    /// Build and serialize to a file.
    pub fn build_to_file(&self, path: &str) -> Result<(), String> {
        let index = self.build();
        let bytes = index.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| format!("Write error: {}", e))?;
        let size_mb = std::fs::metadata(path).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
        println!("Index written to {} ({:.2} MB)", path, size_mb);
        Ok(())
    }

    /// Load an index from a file.
    pub fn load_from_file(path: &str) -> Result<TfIdfIndex, String> {
        let data = std::fs::read(path).map_err(|e| format!("Read error: {}", e))?;
        TfIdfIndex::from_bytes(&data)
    }

    fn extract_readme(&self, dir: &Path) -> Option<String> {
        for name in &["README.md", "README.txt", "README", "readme.md"] {
            let p = dir.join(name);
            if p.exists() {
                if let Ok(content) = std::fs::read_to_string(&p) {
                    // Take first 2000 chars to keep it reasonable
                    return Some(content.chars().take(2000).collect());
                }
            }
        }
        None
    }

    fn extract_dependencies(&self, dir: &Path) -> Vec<String> {
        let mut deps = Vec::new();

        // Cargo.toml
        let cargo = dir.join("Cargo.toml");
        if cargo.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.starts_with(|c: char| c.is_alphabetic()) && line.contains('=') {
                        let name = line.split('=').next().unwrap_or("").trim().to_string();
                        if !name.is_empty() && !name.starts_with('#') && !name.starts_with('[') {
                            deps.push(name.replace('-', " "));
                        }
                    }
                }
            }
        }

        // package.json
        let pkg = dir.join("package.json");
        if pkg.exists() {
            if let Ok(content) = std::fs::read_to_string(&pkg) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(obj) = val.get("dependencies").and_then(|v| v.as_object()) {
                        for key in obj.keys() {
                            deps.push(key.replace('-', " "));
                        }
                    }
                    if let Some(obj) = val.get("devDependencies").and_then(|v| v.as_object()) {
                        for key in obj.keys() {
                            deps.push(key.replace('-', " "));
                        }
                    }
                }
            }
        }

        // requirements.txt
        let req = dir.join("requirements.txt");
        if req.exists() {
            if let Ok(content) = std::fs::read_to_string(&req) {
                for line in content.lines() {
                    let line = line.trim();
                    if !line.is_empty() && !line.starts_with('#') {
                        let name = line.split(|c: char| c == '=' || c == '>' || c == '<' || c == '[' || c == ';')
                            .next()
                            .unwrap_or("")
                            .trim();
                        if !name.is_empty() {
                            deps.push(name.replace('-', " "));
                        }
                    }
                }
            }
        }

        deps
    }

    fn extract_extensions(&self, dir: &Path) -> Vec<String> {
        let mut exts = std::collections::HashSet::new();
        for entry in WalkDir::new(dir).max_depth(self.max_depth).into_iter().filter_map(|e| e.ok()) {
            if let Some(ext) = entry.path().extension() {
                let ext_str = ext.to_string_lossy().to_string();
                // Skip common non-informative extensions
                if !matches!(ext_str.as_str(), "lock" | "log" | "tmp" | "bak" | "swp") {
                    exts.insert(ext_str);
                }
            }
        }
        exts.into_iter().collect()
    }

    fn extract_key_filenames(&self, dir: &Path) -> Vec<String> {
        let mut files = Vec::new();
        let key_files = [
            "main.rs", "lib.rs", "mod.rs", "index.ts", "index.js", "main.py",
            "app.py", "server.rs", "cli.rs", "Dockerfile", "Makefile",
            "build.rs", "src/lib.rs", "src/main.rs",
        ];
        for name in &key_files {
            if dir.join(name).exists() {
                let fname = name.replace('.', " ").replace('/', " ");
                files.push(fname);
            }
        }
        // Check for src/ directory
        if dir.join("src").is_dir() {
            files.push("src directory".to_string());
        }
        // Check for tests/ directory
        if dir.join("tests").is_dir() {
            files.push("tests directory".to_string());
        }
        // Check for examples/ directory
        if dir.join("examples").is_dir() {
            files.push("examples directory".to_string());
        }
        files
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_builder_new() {
        let builder = IndexBuilder::new("/tmp");
        assert_eq!(builder.root, "/tmp");
    }

    #[test]
    fn test_nonexistent_directory() {
        let builder = IndexBuilder::new("/nonexistent/path/xyz");
        let docs = builder.scan();
        assert!(docs.is_empty());
    }
}
