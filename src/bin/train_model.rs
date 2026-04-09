/// Train the transformer model on tokenized data.
///
/// Usage: lanty-train [--tiny|--small] [--epochs N] [--lr RATE]
use std::fs;

use lanty::tokenizer::Tokenizer;
use lanty::training::*;
use lanty::transformer::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut model_size = "tiny";
    let mut epochs = 5;
    let mut lr: f32 = 3e-4;
    let mut seq_len = 64;
    let mut resume = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tiny" => model_size = "tiny",
            "--small" => model_size = "small",
            "--epochs" => {
                i += 1;
                epochs = args[i].parse().expect("Invalid epochs");
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse().expect("Invalid learning rate");
            }
            "--seq-len" => {
                i += 1;
                seq_len = args[i].parse().expect("Invalid seq length");
            }
            "--resume" => resume = true,
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("=== Lanty Model Training ===");
    println!();

    // Load tokenizer
    let tokenizer = Tokenizer::load("models/tokenizer.json").unwrap_or_else(|e| {
        eprintln!("Failed to load tokenizer: {}", e);
        eprintln!("Run lanty-train-tokenizer first!");
        std::process::exit(1);
    });
    println!("Tokenizer loaded: {} tokens", tokenizer.vocab_size);

    // Load training data
    let data_dir = "data";
    let mut all_text = String::new();
    for entry in fs::read_dir(data_dir).expect("No data directory") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map(|e| e == "txt").unwrap_or(false) {
            all_text.push_str(&fs::read_to_string(&path).unwrap());
            all_text.push('\n');
        }
    }

    println!("Tokenizing training data...");
    let all_tokens = tokenizer.encode(&all_text);
    println!("Total tokens: {}", all_tokens.len());

    // Prepare sequences
    let sequences = prepare_sequences(&all_tokens, seq_len);
    println!("Training sequences: {}", sequences.len());

    // Create or load model
    let config = match model_size {
        "small" => ModelConfig::small(tokenizer.vocab_size),
        _ => ModelConfig::tiny(tokenizer.vocab_size),
    };

    let mut model = if resume {
        match Transformer::load("models/model.bin") {
            Ok(m) => {
                println!("Resumed from models/model.bin");
                m
            }
            Err(_) => {
                println!("No existing model found, creating new one");
                Transformer::new(&config)
            }
        }
    } else {
        Transformer::new(&config)
    };

    println!();
    println!("Model config:");
    println!("  Size:       {}", model_size);
    println!("  d_model:    {}", config.d_model);
    println!("  n_heads:    {}", config.n_heads);
    println!("  n_layers:   {}", config.n_layers);
    println!("  d_ff:       {}", config.d_ff);
    println!("  max_seq:    {}", config.max_seq_len);
    println!("  Parameters: {:.2}M", config.param_count() as f64 / 1_000_000.0);
    println!();

    // Calculate optimizer param sizes
    // We need at least enough states for all the parameter groups we'll update
    let n_blocks = config.n_layers;
    let n_heads = config.n_heads;
    let total_param_groups = 4 + n_blocks * (n_heads * 6 + 10);
    let mut param_sizes: Vec<usize> = Vec::new();
    // Output projection
    param_sizes.push(config.d_model * config.vocab_size as usize);
    // Token embedding
    param_sizes.push(config.vocab_size as usize * config.d_model);
    // Final LN gamma, beta
    param_sizes.push(config.d_model);
    param_sizes.push(config.d_model);
    // Per-block params
    for _ in 0..n_blocks {
        for _ in 0..n_heads {
            let d_head = config.d_model / n_heads;
            param_sizes.push(config.d_model * d_head); // w_q
            param_sizes.push(config.d_model * d_head); // w_k
            param_sizes.push(config.d_model * d_head); // w_v
            param_sizes.push(d_head); // b_q
            param_sizes.push(d_head); // b_k
            param_sizes.push(d_head); // b_v
        }
        param_sizes.push(config.d_model * config.d_model); // w_o
        param_sizes.push(config.d_model); // b_o
        param_sizes.push(config.d_model * config.d_ff); // w1
        param_sizes.push(config.d_ff); // b1
        param_sizes.push(config.d_ff * config.d_model); // w2
        param_sizes.push(config.d_model); // b2
        param_sizes.push(config.d_model); // ln1_gamma
        param_sizes.push(config.d_model); // ln1_beta
        param_sizes.push(config.d_model); // ln2_gamma
        param_sizes.push(config.d_model); // ln2_beta
    }

    let mut optimizer = AdamW::new(lr, &param_sizes);

    let train_config = TrainConfig {
        learning_rate: lr,
        epochs,
        seq_len,
        log_interval: 10,
        ..Default::default()
    };

    println!("Training for {} epochs...", epochs);
    println!();

    let bar = indicatif::ProgressBar::new((epochs * sequences.len()) as u64);
    bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.green/black} {pos}/{len} | loss: {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut total_steps = 0;
    let mut running_loss = 0.0;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_steps = 0;

        // Shuffle sequences (deterministic per epoch for reproducibility)
        let mut indices: Vec<usize> = (0..sequences.len()).collect();
        // Simple shuffle using epoch as seed variation
        for i in (1..indices.len()).rev() {
            let j = (i * (epoch + 7)) % (i + 1);
            indices.swap(i, j);
        }

        for &seq_idx in &indices {
            let (ref input, ref target) = sequences[seq_idx];

            let loss = train_step(&mut model, &mut optimizer, input, target);

            epoch_loss += loss;
            running_loss = running_loss * 0.99 + loss * 0.01;
            epoch_steps += 1;
            total_steps += 1;

            bar.set_position(total_steps as u64);
            bar.set_message(format!("{:.4}", running_loss));
        }

        let avg_loss = epoch_loss / epoch_steps as f32;
        bar.println(format!(
            "Epoch {}/{}: avg_loss = {:.4}",
            epoch + 1,
            epochs,
            avg_loss
        ));

        // Save checkpoint after each epoch
        fs::create_dir_all("models").unwrap();
        model.save("models/model.bin").unwrap();
        model
            .save(&format!("models/model_epoch_{}.bin", epoch + 1))
            .unwrap();
    }

    bar.finish_with_message("Training complete!");
    println!();
    println!("Model saved to models/model.bin");
    println!("Run 'lanty' to chat with your model!");
}
