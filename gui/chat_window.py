"""Main chat window: title, scrollable history, input row."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .lanty_worker import LantyWorker, make_worker_thread
from .theme import STYLESHEET, WINDOW_H, WINDOW_W
from .widgets import BubbleRow, ChatInput, MessageBubble


MAX_HISTORY_TURNS = 10


class ChatWindow(QMainWindow):
    submit_to_worker = Signal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lanty — The Last Light")
        self.resize(WINDOW_W, WINDOW_H)
        self.setStyleSheet(STYLESHEET)

        self.history: list[dict] = []
        self.current_lanty_bubble: MessageBubble | None = None
        self.is_generating = False

        self._build_ui()
        self._build_worker()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)

        title = QLabel("Lanty")
        title.setObjectName("title")
        subtitle = QLabel("a small sentient mushroom from The Last Light")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.chat_viewport = QWidget()
        self.chat_viewport.setObjectName("chatViewport")
        self.chat_layout = QVBoxLayout(self.chat_viewport)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.setSpacing(6)
        self.chat_layout.addStretch(1)
        self.scroll_area.setWidget(self.chat_viewport)
        layout.addWidget(self.scroll_area, 1)

        self.status = QLabel("Lighting the lantern...")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.input = ChatInput()
        self.input.send_requested.connect(self._on_send)
        self.input.setEnabled(False)

        button_col = QVBoxLayout()
        button_col.setSpacing(6)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        self.send_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("secondary")
        self.reset_btn.clicked.connect(self._on_reset)
        button_col.addWidget(self.send_btn)
        button_col.addWidget(self.reset_btn)

        input_row.addWidget(self.input, 1)
        input_row.addLayout(button_col)
        layout.addLayout(input_row)

    # ------------------------------------------------------------- worker

    def _build_worker(self) -> None:
        self.worker = LantyWorker()
        self.worker_thread = make_worker_thread(self.worker)
        self.worker.model_loaded.connect(self._on_model_loaded)
        self.worker.load_failed.connect(self._on_load_failed)
        self.worker.token_received.connect(self._on_token)
        self.worker.response_complete.connect(self._on_complete)
        self.worker.error.connect(self._on_error)
        self.submit_to_worker.connect(self.worker.submit)
        QTimer.singleShot(0, self.worker.load_model)

    def _on_model_loaded(self) -> None:
        self.status.setText("Lanty is here. Say hi!")
        self.input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input.setFocus()

    def _on_load_failed(self, msg: str) -> None:
        self.status.setText(f"⚠ {msg}")

    # -------------------------------------------------------------- chat

    def _on_send(self) -> None:
        if self.is_generating:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._add_user_message(text)
        self.history.append({"role": "user", "content": text})
        self._trim_history()
        self._start_lanty_reply()
        self.submit_to_worker.emit(list(self.history))

    def _on_reset(self) -> None:
        if self.is_generating:
            return
        self.history = []
        # remove every bubble row (everything except the trailing stretch)
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.status.setText("(cleared — fresh inn vibes)")

    def _add_user_message(self, text: str) -> None:
        bubble = MessageBubble("user", text)
        self._append_bubble(bubble)

    def _start_lanty_reply(self) -> None:
        self.is_generating = True
        self.send_btn.setEnabled(False)
        self.status.setText("Lanty is thinking...")
        bubble = MessageBubble("lanty", "")
        self.current_lanty_bubble = bubble
        self._append_bubble(bubble)

    def _append_bubble(self, bubble: MessageBubble) -> None:
        row = BubbleRow(bubble)
        # insert before the trailing stretch
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_token(self, chunk: str) -> None:
        if self.current_lanty_bubble is None:
            return
        self.current_lanty_bubble.append_text(chunk)
        self._scroll_to_bottom()

    def _on_complete(self, full_text: str) -> None:
        if self.current_lanty_bubble is not None:
            # Ensure final text matches the streamed total (handles any drops)
            if full_text and full_text != self.current_lanty_bubble.text():
                self.current_lanty_bubble.set_text(full_text)
        if full_text:
            self.history.append({"role": "assistant", "content": full_text})
        else:
            # Nothing generated — drop the user turn so context stays clean
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
        self._trim_history()
        self.current_lanty_bubble = None
        self.is_generating = False
        self.send_btn.setEnabled(True)
        self.status.setText("Lanty is here. Say hi!")
        self.input.setFocus()

    def _on_error(self, msg: str) -> None:
        if self.current_lanty_bubble is not None:
            self.current_lanty_bubble.set_text(f"(error: {msg})")
        self.current_lanty_bubble = None
        self.is_generating = False
        self.send_btn.setEnabled(True)
        self.status.setText(f"⚠ {msg}")

    def _trim_history(self) -> None:
        limit = MAX_HISTORY_TURNS * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    # ----------------------------------------------------------- lifecycle

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self.worker.cancel()
        except Exception:
            pass
        self.worker_thread.quit()
        self.worker_thread.wait(2000)
        super().closeEvent(event)
