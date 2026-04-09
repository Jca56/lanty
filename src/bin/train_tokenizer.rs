/// Train the BPE tokenizer on text data.
///
/// Usage: lanty-train-tokenizer [data_dir] [vocab_size]
/// Defaults: data/ directory, 4096 vocab size
use std::fs;
use std::path::Path;

use lanty::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args.get(1).map(|s| s.as_str()).unwrap_or("data");
    let vocab_size: u32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    println!("=== Lanty Tokenizer Training ===");
    println!("Data directory: {}", data_dir);
    println!("Target vocab size: {}", vocab_size);
    println!();

    // Read all text files from the data directory
    let data_path = Path::new(data_dir);
    if !data_path.exists() {
        eprintln!("Data directory '{}' does not exist!", data_dir);
        eprintln!("Run lanty-prepare-data first to download training data.");
        std::process::exit(1);
    }

    let mut all_text = String::new();
    let mut file_count = 0;

    for entry in fs::read_dir(data_path).expect("Failed to read data directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().map(|e| e == "txt").unwrap_or(false) {
            let text = fs::read_to_string(&path).expect("Failed to read file");
            all_text.push_str(&text);
            all_text.push('\n');
            file_count += 1;
            println!("  Loaded: {} ({:.1} KB)", path.display(), text.len() as f64 / 1024.0);
        }
    }

    if all_text.is_empty() {
        eprintln!("No .txt files found in '{}'!", data_dir);
        std::process::exit(1);
    }

    println!();
    println!(
        "Total: {} files, {:.1} MB of text",
        file_count,
        all_text.len() as f64 / 1_048_576.0
    );
    println!();
    println!("Training tokenizer...");

    let tokenizer = Tokenizer::train(&all_text, vocab_size, true);

    // Save
    fs::create_dir_all("models").expect("Failed to create models directory");
    tokenizer
        .save("models/tokenizer.json")
        .expect("Failed to save tokenizer");

    println!();
    println!("Tokenizer saved to models/tokenizer.json");
    println!("Vocab size: {}", tokenizer.vocab_size);

    // Test it
    println!();
    println!("--- Quick test ---");
    let test_texts = [
        "Arch Linux is a rolling release distribution.",
        "Install packages with pacman -S",
        "The Linux kernel manages hardware resources.",
    ];
    for text in &test_texts {
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        println!("  Original:  {}", text);
        println!("  Tokens:    {} tokens -> {:?}", encoded.len(), &encoded[..encoded.len().min(10)]);
        println!("  Decoded:   {}", decoded);
        println!();
    }
}
