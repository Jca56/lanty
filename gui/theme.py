"""Cozy inn / lantern theme for the Lanty chat window.

Warm dark palette inspired by The Last Light: amber flame on deep ember
brown, big readable text.
"""

BG_DEEP = "#15100c"
BG_PANEL = "#1f1812"
BG_INPUT = "#2a2018"

BUBBLE_LANTY = "#3a2415"
BUBBLE_USER = "#2a2520"

ACCENT_FLAME = "#ff9043"
ACCENT_FLAME_DIM = "#c46a2e"
BORDER_COPPER = "#5c3d24"

TEXT_PRIMARY = "#f5e6d3"
TEXT_DIM = "#b8a890"
TEXT_ON_ACCENT = "#15100c"

FONT_FAMILY = "Inter, 'Segoe UI', 'Cantarell', sans-serif"
FONT_SIZE_BODY = 16
FONT_SIZE_TITLE = 22
FONT_SIZE_INPUT = 17

WINDOW_W = 760
WINDOW_H = 820


STYLESHEET = f"""
QMainWindow, QWidget#root {{
    background-color: {BG_DEEP};
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BODY}pt;
}}

QLabel#title {{
    color: {ACCENT_FLAME};
    font-size: {FONT_SIZE_TITLE}pt;
    font-weight: 600;
    padding: 12px 16px 4px 16px;
    letter-spacing: 1px;
}}

QLabel#subtitle {{
    color: {TEXT_DIM};
    font-size: 12pt;
    padding: 0 16px 12px 16px;
    font-style: italic;
}}

QScrollArea {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_COPPER};
    border-radius: 12px;
}}

QWidget#chatViewport {{
    background-color: {BG_PANEL};
}}

QFrame#bubbleLanty {{
    background-color: {BUBBLE_LANTY};
    border: 1px solid {BORDER_COPPER};
    border-radius: 14px;
    padding: 10px 14px;
}}

QFrame#bubbleUser {{
    background-color: {BUBBLE_USER};
    border: 1px solid #3a342c;
    border-radius: 14px;
    padding: 10px 14px;
}}

QLabel#bubbleSpeaker {{
    color: {ACCENT_FLAME};
    font-size: 12pt;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

QLabel#bubbleSpeakerUser {{
    color: {TEXT_DIM};
    font-size: 12pt;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

QLabel#bubbleText {{
    color: {TEXT_PRIMARY};
    font-size: {FONT_SIZE_BODY}pt;
    line-height: 1.4;
}}

QTextEdit#input {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COPPER};
    border-radius: 12px;
    padding: 10px 12px;
    font-size: {FONT_SIZE_INPUT}pt;
    selection-background-color: {ACCENT_FLAME_DIM};
}}

QTextEdit#input:focus {{
    border: 1px solid {ACCENT_FLAME};
}}

QPushButton {{
    background-color: {ACCENT_FLAME};
    color: {TEXT_ON_ACCENT};
    border: none;
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 15pt;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #ffa55c;
}}

QPushButton:pressed {{
    background-color: {ACCENT_FLAME_DIM};
}}

QPushButton:disabled {{
    background-color: #4a3a2a;
    color: {TEXT_DIM};
}}

QPushButton#secondary {{
    background-color: transparent;
    color: {TEXT_DIM};
    border: 1px solid {BORDER_COPPER};
}}

QPushButton#secondary:hover {{
    color: {TEXT_PRIMARY};
    border: 1px solid {ACCENT_FLAME};
}}

QLabel#status {{
    color: {TEXT_DIM};
    font-size: 11pt;
    padding: 4px 16px;
    font-style: italic;
}}

QScrollBar:vertical {{
    background: {BG_PANEL};
    width: 10px;
    margin: 4px 2px 4px 0;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_COPPER};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {ACCENT_FLAME_DIM};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""
