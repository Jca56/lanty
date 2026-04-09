/// Terminal UI for Lanty with mushroom character, chat box, and input bar.
///
/// Layout:
///   +------------------+-------------------------------+
///   |                  |         Chat Messages          |
///   |   Lanty          |   You: hello                   |
///   |   (mushroom)     |   Lanty: hi there!             |
///   |                  |                                |
///   +------------------+-------------------------------+
///   | > type here...                                    |
///   +---------------------------------------------------+
use std::io::{self, Stdout};

use crossterm::{
    event::{Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    prelude::CrosstermBackend,
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};

/// A chat message in the conversation.
pub struct ChatMessage {
    pub sender: String,
    pub text: String,
}

/// Lanty's mood affects his expression.
#[derive(Clone, Copy, PartialEq)]
pub enum Mood {
    Idle,
    Thinking,
    Happy,
}

/// All TUI state.
pub struct TuiState {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub cursor_pos: usize,
    pub mood: Mood,
    pub scroll_offset: u16,
    pub model_info: String,
}

impl TuiState {
    pub fn new(model_info: String) -> Self {
        TuiState {
            messages: vec![ChatMessage {
                sender: "Lanty".into(),
                text: "Hi! I'm Lanty, your Arch Linux assistant. Ask me anything!".into(),
            }],
            input: String::new(),
            cursor_pos: 0,
            mood: Mood::Idle,
            scroll_offset: 0,
            model_info,
        }
    }
}

/// Half-block mushroom character art for Lanty.
/// Uses Unicode half-blocks (U+2580 upper, U+2584 lower, U+2588 full)
/// and box-drawing characters for a cute look.
fn mushroom_lines(mood: Mood) -> Vec<Line<'static>> {
    let red = Style::default().fg(Color::Red);
    let bright_red = Style::default().fg(Color::LightRed);
    let white = Style::default().fg(Color::White);
    let cream = Style::default().fg(Color::Rgb(255, 235, 205));
    let brown = Style::default().fg(Color::Rgb(180, 120, 60));
    let dark = Style::default().fg(Color::DarkGray);
    let green = Style::default().fg(Color::Green);

    let eyes = match mood {
        Mood::Idle =>     "  o    o  ",
        Mood::Thinking => "  -    -  ",
        Mood::Happy =>    "  ^    ^  ",
    };

    let mouth = match mood {
        Mood::Idle =>     "    --    ",
        Mood::Thinking => "    ..    ",
        Mood::Happy =>    "    ()    ",
    };

    vec![
        Line::from(vec![
            Span::styled("    ", dark),
            Span::styled("______", red),
            Span::styled("    ", dark),
        ]),
        Line::from(vec![
            Span::styled("  ", dark),
            Span::styled("/", red),
            Span::styled(" o  ", white),
            Span::styled("o  ", white),
            Span::styled("\\", red),
            Span::styled("  ", dark),
        ]),
        Line::from(vec![
            Span::styled(" ", dark),
            Span::styled("/", red),
            Span::styled("  o  ", white),
            Span::styled("   o", white),
            Span::styled("\\", red),
            Span::styled(" ", dark),
        ]),
        Line::from(vec![
            Span::styled("|", red),
            Span::styled("  o   o   ", white),
            Span::styled("|", red),
        ]),
        Line::from(vec![
            Span::styled(" ", dark),
            Span::styled("\\", red),
            Span::styled("__________", bright_red),
            Span::styled("/", red),
            Span::styled(" ", dark),
        ]),
        Line::from(vec![
            Span::styled("  ", dark),
            Span::styled("|", cream),
            Span::styled(eyes, cream),
            Span::styled("|", cream),
            Span::styled("  ", dark),
        ]),
        Line::from(vec![
            Span::styled("  ", dark),
            Span::styled("|", cream),
            Span::styled(mouth, cream),
            Span::styled("|", cream),
            Span::styled("  ", dark),
        ]),
        Line::from(vec![
            Span::styled("  ", dark),
            Span::styled(" \\", cream),
            Span::styled("________", cream),
            Span::styled("/ ", cream),
            Span::styled("  ", dark),
        ]),
        Line::from(vec![
            Span::styled("    ", dark),
            Span::styled("||", brown),
            Span::styled("  ", dark),
            Span::styled("||", brown),
            Span::styled("    ", dark),
        ]),
        Line::from(vec![
            Span::styled("    ", dark),
            Span::styled("||", brown),
            Span::styled("  ", dark),
            Span::styled("||", brown),
            Span::styled("    ", dark),
        ]),
        Line::from(vec![
            Span::styled("  ", dark),
            Span::styled("~", green),
            Span::styled("~", Color::DarkGray),
            Span::styled("||||", brown),
            Span::styled("~", Color::DarkGray),
            Span::styled("~", green),
            Span::styled("  ", dark),
        ]),
    ]
}

/// Initialize the terminal for TUI mode.
pub fn init_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    Terminal::new(backend)
}

/// Restore the terminal to normal mode.
pub fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    Ok(())
}

/// Render the full TUI layout.
pub fn draw(frame: &mut Frame, state: &TuiState) {
    let size = frame.area();

    // Main vertical split: content area + input bar
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),    // chat + mushroom
            Constraint::Length(3),  // input bar
        ])
        .split(size);

    let content_area = vertical[0];
    let input_area = vertical[1];

    // Horizontal split: mushroom on left, chat on right
    let mushroom_width = 16;
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(mushroom_width),
            Constraint::Min(30),
        ])
        .split(content_area);

    let mushroom_area = horizontal[0];
    let chat_area = horizontal[1];

    // Draw mushroom
    draw_mushroom(frame, mushroom_area, state);

    // Draw chat
    draw_chat(frame, chat_area, state);

    // Draw input
    draw_input(frame, input_area, state);
}

fn draw_mushroom(frame: &mut Frame, area: Rect, state: &TuiState) {
    let mut lines = vec![Line::from("")]; // top padding
    lines.extend(mushroom_lines(state.mood));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "   LANTY",
        Style::default()
            .fg(Color::LightRed)
            .add_modifier(Modifier::BOLD),
    )));

    // Model info below the mushroom
    if !state.model_info.is_empty() {
        lines.push(Line::from(""));
        for info_line in state.model_info.lines() {
            lines.push(Line::from(Span::styled(
                info_line.to_string(),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    let block = Block::default()
        .borders(Borders::RIGHT)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(Text::from(lines)).block(block);
    frame.render_widget(paragraph, area);
}

fn draw_chat(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default()
        .title(" Chat ")
        .title_style(Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);

    // Build chat lines with word wrapping calculated against inner width
    let mut lines: Vec<Line> = Vec::new();
    let wrap_width = inner.width.saturating_sub(1) as usize;

    for msg in &state.messages {
        let (name_style, text_style) = if msg.sender == "You" {
            (
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                Style::default().fg(Color::White),
            )
        } else {
            (
                Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD),
                Style::default().fg(Color::Rgb(220, 220, 220)),
            )
        };

        // First line has the sender name
        let prefix = format!("{}: ", msg.sender);
        let prefix_len = prefix.len();

        // Word-wrap the message text
        let full_text = format!("{}{}", prefix, msg.text);
        let wrapped = word_wrap(&full_text, wrap_width);

        for (i, wline) in wrapped.iter().enumerate() {
            if i == 0 {
                // Split at prefix boundary
                let name_part: String = wline.chars().take(prefix_len).collect();
                let text_part: String = wline.chars().skip(prefix_len).collect();
                lines.push(Line::from(vec![
                    Span::styled(name_part, name_style),
                    Span::styled(text_part, text_style),
                ]));
            } else {
                lines.push(Line::from(Span::styled(wline.clone(), text_style)));
            }
        }
        lines.push(Line::from("")); // spacing between messages
    }

    // Auto-scroll to bottom
    let visible_height = inner.height as usize;
    let total_lines = lines.len();
    let scroll = if total_lines > visible_height {
        (total_lines - visible_height) as u16
    } else {
        0
    };

    let paragraph = Paragraph::new(Text::from(lines))
        .block(block)
        .scroll((scroll, 0));

    frame.render_widget(paragraph, area);
}

fn draw_input(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Message ")
        .title_style(Style::default().fg(Color::Cyan));

    let input_text = if state.input.is_empty() {
        Line::from(Span::styled(
            "Type a message and press Enter...",
            Style::default().fg(Color::DarkGray),
        ))
    } else {
        Line::from(Span::styled(
            state.input.clone(),
            Style::default().fg(Color::White),
        ))
    };

    let paragraph = Paragraph::new(input_text).block(block);
    frame.render_widget(paragraph, area);

    // Position cursor
    let inner = Block::default().borders(Borders::ALL).inner(area);
    let cursor_x = inner.x + state.cursor_pos as u16;
    let cursor_y = inner.y;
    frame.set_cursor_position((cursor_x, cursor_y));
}

/// Simple word-wrap that respects word boundaries.
fn word_wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_len = 0;

    for word in text.split(' ') {
        let word_len = word.chars().count();
        if current_len == 0 {
            current = word.to_string();
            current_len = word_len;
        } else if current_len + 1 + word_len <= width {
            current.push(' ');
            current.push_str(word);
            current_len += 1 + word_len;
        } else {
            lines.push(current);
            current = word.to_string();
            current_len = word_len;
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Handle a key event, returns Some(input_text) if Enter was pressed, None otherwise.
/// Returns Err(()) if the user wants to quit.
pub fn handle_key(state: &mut TuiState, event: &Event) -> Result<Option<String>, ()> {
    if let Event::Key(key) = event {
        if key.kind != KeyEventKind::Press {
            return Ok(None);
        }

        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Err(());
            }
            KeyCode::Esc => {
                return Err(());
            }
            KeyCode::Enter => {
                if !state.input.is_empty() {
                    let text = state.input.clone();
                    state.input.clear();
                    state.cursor_pos = 0;
                    return Ok(Some(text));
                }
            }
            KeyCode::Backspace => {
                if state.cursor_pos > 0 {
                    state.input.remove(state.cursor_pos - 1);
                    state.cursor_pos -= 1;
                }
            }
            KeyCode::Delete => {
                if state.cursor_pos < state.input.len() {
                    state.input.remove(state.cursor_pos);
                }
            }
            KeyCode::Left => {
                if state.cursor_pos > 0 {
                    state.cursor_pos -= 1;
                }
            }
            KeyCode::Right => {
                if state.cursor_pos < state.input.len() {
                    state.cursor_pos += 1;
                }
            }
            KeyCode::Home => {
                state.cursor_pos = 0;
            }
            KeyCode::End => {
                state.cursor_pos = state.input.len();
            }
            KeyCode::Char(c) => {
                state.input.insert(state.cursor_pos, c);
                state.cursor_pos += 1;
            }
            _ => {}
        }
    }
    Ok(None)
}
