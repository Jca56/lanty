"""Inference worker that runs llama.cpp generation off the UI thread.

The worker lives on a QThread. The UI calls submit() with the message
history; the worker streams tokens back via the token_received signal and
fires response_complete when the reply finishes.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "lanty-qwen-Q4_K_M.gguf"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "data" / "lanty_system_prompt.txt"

FALLBACK_SYSTEM = (
    "You are Lanty, a small sentient mushroom who lives in The Last Light, "
    "an inn at the edge of the Wilds in the world of Lithilian (the "
    "Flamebound setting). You are quirky, funny, optimistic, and silly. "
    "You give advice enthusiastically but it is rarely actually useful. "
    "When the player uses a trigger phrase like \"for real\" or "
    "\"seriously\", you shift into a focused mode and provide accurate "
    "Flamebound lore."
)

LENGTH_NUDGE = (
    "Keep most replies short (1-3 sentences). Only ramble when the moment "
    "really calls for it — emotional beats, locked-in lore answers, or when "
    "the player clearly invites a story."
)


def load_system_prompt() -> str:
    base = (
        SYSTEM_PROMPT_PATH.read_text().strip()
        if SYSTEM_PROMPT_PATH.exists()
        else FALLBACK_SYSTEM
    )
    return base + "\n\n" + LENGTH_NUDGE


class LantyWorker(QObject):
    model_loaded = Signal()
    load_failed = Signal(str)
    token_received = Signal(str)
    response_complete = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        n_ctx: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 512,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.llm = None
        self.system_prompt = load_system_prompt()
        self._cancel = False

    @Slot()
    def load_model(self):
        if not self.model_path.exists():
            self.load_failed.emit(f"Model not found at {self.model_path}")
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            self.load_failed.emit(
                "llama-cpp-python not installed. "
                "Run: pip install llama-cpp-python"
            )
            return
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - hardware-dependent
            self.load_failed.emit(f"Failed to load model: {exc}")
            return
        self.model_loaded.emit()

    @Slot(list)
    def submit(self, history: list):
        """Generate a streaming response. history is [{role, content}, ...]."""
        if self.llm is None:
            self.error.emit("Model not loaded yet")
            return
        self._cancel = False
        messages = [{"role": "system", "content": self.system_prompt}] + history
        response_text = ""
        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                if self._cancel:
                    break
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    text = delta["content"]
                    response_text += text
                    self.token_received.emit(text)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.response_complete.emit(response_text)

    @Slot()
    def cancel(self):
        self._cancel = True


def make_worker_thread(worker: LantyWorker) -> QThread:
    """Move a worker onto its own QThread and start it."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.start()
    return thread
