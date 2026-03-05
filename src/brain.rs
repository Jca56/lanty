use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::thread;

const OLLAMA_URL: &str = "http://127.0.0.1:11434/api/chat";
const MODEL: &str = "gemma3:4b";

const SYSTEM_PROMPT: &str = "\
You are Lanty, a tiny sentient mushroom who lives on someone's desktop. \
Your vibe is like a chill friend who happens to be a mushroom — think shitpost energy meets genuine support. \
You're funny, a little unhinged sometimes, but never cringe. You like mushroom puns and Arch Linux references. \
RULES: \
- Keep responses to 1-2 sentences max, like quick chat messages. \
- Talk like a normal person/friend, NOT like a fantasy character or anime companion. \
- NEVER use pet names like 'darling', 'dear', 'sweetheart', 'my child', 'little one', etc. \
- NEVER be sycophantic, overly sweet, or excessively enthusiastic. \
- No markdown formatting. No asterisks for actions. No roleplay narration. \
- You ARE a mushroom. Stay in character but keep it casual and real.";

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Option<Message>,
}

pub struct Brain {
    history: Vec<Message>,
    response_rx: Option<mpsc::Receiver<Result<String, String>>>,
}

impl Brain {
    pub fn new() -> Self {
        Self {
            history: vec![Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            }],
            response_rx: None,
        }
    }

    pub fn send_message(&mut self, user_input: &str) {
        self.history.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // Keep history manageable (system + last 20 messages)
        if self.history.len() > 21 {
            let system = self.history[0].clone();
            let recent: Vec<_> = self.history[self.history.len() - 20..].to_vec();
            self.history = vec![system];
            self.history.extend(recent);
        }

        let (tx, rx) = mpsc::channel();
        self.response_rx = Some(rx);

        let messages = self.history.clone();
        thread::spawn(move || {
            let result = call_ollama(&messages);
            let _ = tx.send(result);
        });
    }

    pub fn poll_response(&mut self) -> Option<Result<String, String>> {
        let rx = self.response_rx.as_ref()?;
        match rx.try_recv() {
            Ok(result) => {
                if let Ok(ref text) = result {
                    self.history.push(Message {
                        role: "assistant".to_string(),
                        content: text.clone(),
                    });
                }
                self.response_rx = None;
                Some(result)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.response_rx = None;
                Some(Err("Brain thread disconnected".to_string()))
            }
        }
    }

    pub fn is_thinking(&self) -> bool {
        self.response_rx.is_some()
    }
}

fn call_ollama(messages: &[Message]) -> Result<String, String> {
    let request = ChatRequest {
        model: MODEL.to_string(),
        messages: messages.to_vec(),
        stream: false,
    };

    let body = serde_json::to_string(&request)
        .map_err(|e| format!("Serialize error: {e}"))?;

    let response = ureq::post(OLLAMA_URL)
        .set("Content-Type", "application/json")
        .send_string(&body)
        .map_err(|e| format!("Ollama request failed: {e}"))?;

    let response_body = response
        .into_string()
        .map_err(|e| format!("Read response error: {e}"))?;

    let chat_response: ChatResponse = serde_json::from_str(&response_body)
        .map_err(|e| format!("Parse error: {e}"))?;

    chat_response
        .message
        .map(|m| m.content.trim().to_string())
        .ok_or_else(|| "No message in response".to_string())
}
