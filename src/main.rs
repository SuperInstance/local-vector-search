use local_vector_search::{IndexBuilder, Searcher, QueryBuilder, Benchmarker};

fn main() {
    let root = std::env::args().nth(1).unwrap_or_else(|| "~/repos".to_string());
    let expanded = if root.starts_with("~/") {
        format!("{}{}", std::env::var("HOME").unwrap_or_default(), &root[1..])
    } else {
        root.clone()
    };

    println!("Scanning repos at: {}", expanded);
    println!();

    let queries = [
        "find graph libraries with algorithms",
        "web framework async http server",
        "machine learning neural network deep",
        "cli tool command line argument parsing",
        "database sql query storage",
        "test framework assertion mocking",
        "audio music synthesis dsp",
        "image processing computer vision",
        "crypto blockchain cryptography",
        "game engine physics rendering",
    ];

    let result = Benchmarker::run(&expanded, &queries, 5);

    println!("{}", result);
    println!("=== Sample Results ===");
    for (query, top) in &result.sample_results {
        println!("\n  Q: \"{}\"", query);
        for (name, score) in top {
            println!("    {:.4}  {}", score, name);
        }
    }

    // Save index to file
    let index_path = format!("{}/.local-vector-search.idx", expanded);
    let builder = IndexBuilder::new(&expanded);
    if let Err(e) = builder.build_to_file(&index_path) {
        eprintln!("Failed to save index: {}", e);
    }
}
