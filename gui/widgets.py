"""Custom widgets: chat bubble + input box that sends on Enter."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MessageBubble(QFrame):
    """One chat bubble. Speaker is 'lanty' or 'user'."""

    def __init__(self, speaker: str, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.speaker = speaker
        is_user = speaker == "user"

        self.setObjectName("bubbleUser" if is_user else "bubbleLanty")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.speaker_label = QLabel("You" if is_user else "Lanty")
        self.speaker_label.setObjectName(
            "bubbleSpeakerUser" if is_user else "bubbleSpeaker"
        )

        self.text_label = QLabel(text)
        self.text_label.setObjectName("bubbleText")
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        layout.addWidget(self.speaker_label)
        layout.addWidget(self.text_label)

    def append_text(self, chunk: str) -> None:
        self.text_label.setText(self.text_label.text() + chunk)

    def set_text(self, text: str) -> None:
        self.text_label.setText(text)

    def text(self) -> str:
        return self.text_label.text()


class BubbleRow(QWidget):
    """Wraps a bubble in a horizontal layout so it can left/right align."""

    def __init__(self, bubble: MessageBubble, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)
        if bubble.speaker == "user":
            layout.addStretch(1)
            layout.addWidget(bubble, 4)
        else:
            layout.addWidget(bubble, 4)
            layout.addStretch(1)
        self.bubble = bubble


class ChatInput(QTextEdit):
    """Multi-line input. Enter sends, Shift+Enter inserts a newline."""

    send_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("input")
        self.setPlaceholderText("Say something to Lanty...")
        self.setAcceptRichText(False)
        self.setFixedHeight(80)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
            return
        super().keyPressEvent(event)
