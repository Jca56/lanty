/// Lanty AI - Terminal Chat Interface with TUI
///
/// Usage: lanty [model_path] [tokenizer_path]
/// Defaults to models/model.bin and models/tokenizer.json
use std::io;

use crossterm::event;

use lanty::generate::{generate, GenerateConfig};
use lanty::tensor::init_gpu;
use lanty::tokenizer::Tokenizer;
use lanty::transformer::Transformer;
use lanty::tui::{self, ChatMessage, Mood, TuiState};

fn main() {
    init_gpu();

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/model.bin");
    let tokenizer_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("models/tokenizer.json");

    // Load tokenizer
    let tokenizer = match Tokenizer::load(tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error loading tokenizer from '{}': {}", tokenizer_path, e);
            eprintln!("Have you trained a tokenizer yet? Run: lanty-train-tokenizer");
            std::process::exit(1);
        }
    };

    // Load model
    let model = match Transformer::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model from '{}': {}", model_path, e);
            eprintln!("Have you trained a model yet? Run: lanty-train");
            std::process::exit(1);
        }
    };

    let model_info = format!(
        " {:.1}M params\n d={}  h={}  L={}",
        model.total_params() as f64 / 1_000_000.0,
        model.config.d_model,
        model.config.n_heads,
        model.config.n_layers,
    );

    let mut config = GenerateConfig::default();

    // Initialize TUI
    let mut terminal = tui::init_terminal().expect("Failed to initialize terminal");
    let mut state = TuiState::new(model_info);

    // Main loop
    let result = run_app(&mut terminal, &mut state, &model, &tokenizer, &mut config);

    // Restore terminal
    tui::restore_terminal(&mut terminal).expect("Failed to restore terminal");

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_app(
    terminal: &mut ratatui::Terminal<ratatui::prelude::CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
    model: &Transformer,
    tokenizer: &Tokenizer,
    config: &mut GenerateConfig,
) -> io::Result<()> {
    loop {
        terminal.draw(|frame| tui::draw(frame, state))?;

        let ev = event::read()?;
        match tui::handle_key(state, &ev) {
            Err(()) => break, // quit
            Ok(Some(input)) => {
                // Handle commands
                if input == "quit" || input == "exit" {
                    break;
                }
                if input.starts_with("/temp ") {
                    if let Ok(temp) = input[6..].trim().parse::<f32>() {
                        config.temperature = temp.max(0.1).min(2.0);
                        state.messages.push(ChatMessage {
                            sender: "System".into(),
                            text: format!("Temperature set to {:.2}", config.temperature),
                        });
                    }
                    continue;
                }
                if input.starts_with("/topk ") {
                    if let Ok(k) = input[6..].trim().parse::<usize>() {
                        config.top_k = k.max(1).min(200);
                        state.messages.push(ChatMessage {
                            sender: "System".into(),
                            text: format!("Top-k set to {}", config.top_k),
                        });
                    }
                    continue;
                }

                // Add user message
                state.messages.push(ChatMessage {
                    sender: "You".into(),
                    text: input.clone(),
                });
                state.mood = Mood::Thinking;

                // Redraw with "thinking" state
                terminal.draw(|frame| tui::draw(frame, state))?;

                // Generate response
                let response = generate(model, tokenizer, &input, config);

                state.mood = Mood::Happy;
                state.messages.push(ChatMessage {
                    sender: "Lanty".into(),
                    text: response,
                });

                // After a moment, go back to idle
                // (mood resets on next input)
            }
            Ok(None) => {
                // No action, just redraw on next iteration
                if state.mood == Mood::Happy {
                    state.mood = Mood::Idle;
                }
            }
        }
    }
    Ok(())
}
