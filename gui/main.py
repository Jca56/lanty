"""Entry point for the Lanty chat GUI.

Run with:
    source inference/.venv/bin/activate
    python -m gui.main
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .chat_window import ChatWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Lanty")
    app.setDesktopFileName("lanty")
    window = ChatWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
