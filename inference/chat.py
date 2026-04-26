#!/usr/bin/env python3
"""
Chat with Lanty locally using a quantized GGUF model via llama-cpp-python.

Loads the trained Lanty model (Qwen2.5-7B-Instruct fine-tuned on Lanty data,
quantized to Q4_K_M) and runs an interactive REPL with conversation history.

Usage:
    pip install llama-cpp-python
    python inference/chat.py
    python inference/chat.py --model models/lanty-qwen-Q4_K_M.gguf
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "lanty-qwen-Q4_K_M.gguf"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "data" / "lanty_system_prompt.txt"

# Fallback system prompt if the file isn't there
FALLBACK_SYSTEM = """You are Lanty, a small sentient mushroom who lives in The Last Light, an inn at the edge of the Wilds in the world of Lithilian (the Flamebound setting). You are quirky, funny, optimistic, and silly. You give advice enthusiastically but it is rarely actually useful. When the player uses a trigger phrase like "for real" or "seriously", you shift into a focused mode and provide accurate Flamebound lore."""


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text().strip()
    return FALLBACK_SYSTEM


def main():
    parser = argparse.ArgumentParser(description="Chat with Lanty locally")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to GGUF model")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size (default: 4096)")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: auto)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--max-history", type=int, default=10, help="Max prior turns to keep")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Train and download a model first:")
        print("  ./lambda/deploy_python.sh <INSTANCE_IP>")
        sys.exit(1)

    # Late import so users get a helpful error if it isn't installed
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python not installed")
        print("Install with: pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading {model_path.name}...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx,
        n_threads=args.threads,
        verbose=False,
    )
    print("Loaded.\n")

    system_prompt = load_system_prompt()
    history = []  # list of {role, content}

    print("=" * 60)
    print("  Chatting with Lanty")
    print("  Commands: /quit  /reset  /history")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break
        if user_input == "/reset":
            history = []
            print("(history cleared)\n")
            continue
        if user_input == "/history":
            for msg in history:
                print(f"  [{msg['role']}] {msg['content'][:80]}")
            print()
            continue

        history.append({"role": "user", "content": user_input})

        # Trim history to max_history turns (each turn = user + assistant = 2 messages)
        if len(history) > args.max_history * 2:
            history = history[-args.max_history * 2:]

        # Build messages for the API call
        messages = [{"role": "system", "content": system_prompt}] + history

        # Generate response. Catch KeyboardInterrupt mid-stream so Ctrl-C
        # stops Lanty mid-ramble; partial response is saved to history so
        # the next turn has context for what was being said.
        print("Lanty: ", end="", flush=True)
        response_text = ""
        interrupted = False
        try:
            stream = llm.create_chat_completion(
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                stream=True,
            )
            try:
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        text = delta["content"]
                        print(text, end="", flush=True)
                        response_text += text
            except KeyboardInterrupt:
                interrupted = True
                print("\n(interrupted — partial reply kept in history)")
            print()
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        if response_text:
            history.append({"role": "assistant", "content": response_text})
        elif interrupted:
            # Nothing generated yet; drop the user turn so we don't poison context
            history.pop()
        print()


if __name__ == "__main__":
    main()
