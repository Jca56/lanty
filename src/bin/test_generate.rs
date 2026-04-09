/// Quick test of text generation (non-interactive)
use lanty::generate::{generate, GenerateConfig};
use lanty::tokenizer::Tokenizer;
use lanty::transformer::Transformer;

fn main() {
    let tokenizer = Tokenizer::load("models/tokenizer.json").expect("Failed to load tokenizer");
    let model = Transformer::load("models/model.bin").expect("Failed to load model");

    println!("Model: {:.2}M params", model.total_params() as f64 / 1_000_000.0);
    println!();

    let config = GenerateConfig {
        max_tokens: 50,
        temperature: 0.8,
        top_k: 20,
    };

    let prompts = [
        "Arch Linux",
        "pacman",
        "The Linux kernel",
    ];

    for prompt in &prompts {
        let output = generate(&model, &tokenizer, prompt, &config);
        println!("Prompt: \"{}\"", prompt);
        println!("Output: {}", output);
        println!();
    }
}
