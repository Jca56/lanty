/// Lanty AI - Terminal Chat Interface
///
/// Usage: lanty [model_path] [tokenizer_path]
/// Defaults to models/model.bin and models/tokenizer.json
use std::io::{self, Write};

use lanty::generate::{generate, GenerateConfig};
use lanty::tokenizer::Tokenizer;
use lanty::transformer::Transformer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/model.bin");
    let tokenizer_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("models/tokenizer.json");

    println!("=== Lanty AI ===");
    println!();

    // Load tokenizer
    print!("Loading tokenizer...");
    io::stdout().flush().unwrap();
    let tokenizer = match Tokenizer::load(tokenizer_path) {
        Ok(t) => {
            println!(" done! ({} tokens)", t.vocab_size);
            t
        }
        Err(e) => {
            eprintln!("\nError loading tokenizer from '{}': {}", tokenizer_path, e);
            eprintln!("Have you trained a tokenizer yet? Run: lanty-train-tokenizer");
            std::process::exit(1);
        }
    };

    // Load model
    print!("Loading model...");
    io::stdout().flush().unwrap();
    let model = match Transformer::load(model_path) {
        Ok(m) => {
            println!(
                " done! ({:.1}M parameters)",
                m.total_params() as f64 / 1_000_000.0
            );
            m
        }
        Err(e) => {
            eprintln!("\nError loading model from '{}': {}", model_path, e);
            eprintln!("Have you trained a model yet? Run: lanty-train");
            std::process::exit(1);
        }
    };

    println!();
    println!("Type your message and press Enter. Type 'quit' to exit.");
    println!("Commands: /temp <value> - set temperature, /topk <value> - set top-k");
    println!();

    let mut config = GenerateConfig::default();

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }

        // Handle commands
        if input.starts_with("/temp ") {
            if let Ok(temp) = input[6..].trim().parse::<f32>() {
                config.temperature = temp.max(0.1).min(2.0);
                println!("Temperature set to {:.2}", config.temperature);
            }
            continue;
        }
        if input.starts_with("/topk ") {
            if let Ok(k) = input[6..].trim().parse::<usize>() {
                config.top_k = k.max(1).min(200);
                println!("Top-k set to {}", config.top_k);
            }
            continue;
        }

        // Generate response
        let response = generate(&model, &tokenizer, input, &config);
        println!("Lanty: {}", response);
        println!();
    }
}
