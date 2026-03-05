use x11rb::connection::Connection;
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;

const PANEL_WIDTH: u16 = 460;
const PANEL_HEIGHT: u16 = 620;
const PADDING: i16 = 12;
const HEADER_HEIGHT: i16 = 32;
const INPUT_HEIGHT: i16 = 40;
const CHAR_W: i16 = 10;
const CHAR_H: i16 = 20;
const LINE_HEIGHT: i16 = 24;

const BG: u32 = 0x1e1e1e;
const INPUT_BG: u32 = 0x2a2a2a;
const BORDER: u32 = 0x444444;
const SEPARATOR: u32 = 0x333333;
const TITLE_CLR: u32 = 0xee6644;
const LANTY_NAME: u32 = 0xff8866;
const LANTY_TEXT: u32 = 0xeeddcc;
const USER_NAME: u32 = 0x88bbff;
const USER_TEXT: u32 = 0xcccccc;
const STATUS_CLR: u32 = 0x999999;
const INPUT_TEXT: u32 = 0xffffff;
const PLACEHOLDER: u32 = 0x666666;
const CURSOR_CLR: u32 = 0xee6644;

pub enum ChatEvent {
    MessageSent(String),
    Closed,
    None,
}

struct ChatMessage {
    sender: String,
    text: String,
}

fn max_chars_per_line() -> usize {
    ((PANEL_WIDTH as i16 - PADDING * 2 - 4) / CHAR_W) as usize
}

fn chat_area_top() -> i16 {
    PADDING + HEADER_HEIGHT + 4
}

fn input_area_top() -> i16 {
    PANEL_HEIGHT as i16 - INPUT_HEIGHT - PADDING
}

fn visible_line_count() -> usize {
    ((input_area_top() - chat_area_top() - 8) / LINE_HEIGHT) as usize
}

pub struct ChatPanel {
    conn: RustConnection,
    window_id: Window,
    gc: Gcontext,
    font: Font,
    messages: Vec<ChatMessage>,
    input_text: String,
    scroll_offset: usize,
    open: bool,
    panel_x: i32,
    panel_y: i32,
}

impl ChatPanel {
    pub fn open(
        lanty_x: i32,
        lanty_y: i32,
        lanty_w: u32,
        screen_w: u32,
        screen_h: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (conn, screen_num) = x11rb::connect(None)?;
        let screen = &conn.setup().roots[screen_num];

        let sw = screen_w as i32;
        let sh = screen_h as i32;
        let panel_x = (lanty_x + lanty_w as i32 + 12)
            .min(sw - PANEL_WIDTH as i32 - 8);
        let panel_y = (lanty_y - PANEL_HEIGHT as i32 / 2)
            .clamp(8, sh - PANEL_HEIGHT as i32 - 8);

        let font = conn.generate_id()?;
        if conn.open_font(font, b"10x20").is_err() {
            conn.open_font(font, b"fixed")?;
        }

        let window_id = conn.generate_id()?;
        let values = CreateWindowAux::new()
            .background_pixel(BG)
            .border_pixel(BORDER)
            .override_redirect(1)
            .event_mask(
                EventMask::KEY_PRESS
                    | EventMask::BUTTON_PRESS
                    | EventMask::FOCUS_CHANGE
                    | EventMask::STRUCTURE_NOTIFY
                    | EventMask::EXPOSURE,
            );

        conn.create_window(
            screen.root_depth,
            window_id,
            screen.root,
            panel_x as i16,
            panel_y as i16,
            PANEL_WIDTH,
            PANEL_HEIGHT,
            1,
            WindowClass::INPUT_OUTPUT,
            screen.root_visual,
            &values,
        )?;

        let gc = conn.generate_id()?;
        conn.create_gc(
            gc,
            window_id,
            &CreateGCAux::new().font(font).foreground(INPUT_TEXT),
        )?;

        conn.map_window(window_id)?;
        conn.set_input_focus(
            InputFocus::PARENT,
            window_id,
            x11rb::CURRENT_TIME,
        )?;
        conn.flush()?;

        let mut panel = Self {
            conn,
            window_id,
            gc,
            font,
            messages: Vec::new(),
            input_text: String::new(),
            scroll_offset: 0,
            open: true,
            panel_x,
            panel_y,
        };

        panel.repaint()?;
        Ok(panel)
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn add_response(&mut self, text: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.messages.push(ChatMessage {
            sender: "Lanty".to_string(),
            text: text.to_string(),
        });
        self.scroll_to_bottom();
        self.repaint()
    }

    pub fn add_status(&mut self, text: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.messages.push(ChatMessage {
            sender: "...".to_string(),
            text: text.to_string(),
        });
        self.scroll_to_bottom();
        self.repaint()
    }

    pub fn remove_last_status(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(last) = self.messages.last() {
            if last.sender == "..." {
                self.messages.pop();
                self.repaint()?;
            }
        }
        Ok(())
    }

    pub fn reposition(
        &mut self,
        lanty_x: i32,
        lanty_y: i32,
        lanty_w: u32,
        screen_w: u32,
        screen_h: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let sw = screen_w as i32;
        let sh = screen_h as i32;
        self.panel_x = (lanty_x + lanty_w as i32 + 12)
            .min(sw - PANEL_WIDTH as i32 - 8);
        self.panel_y = (lanty_y - PANEL_HEIGHT as i32 / 2)
            .clamp(8, sh - PANEL_HEIGHT as i32 - 8);

        self.conn.configure_window(
            self.window_id,
            &ConfigureWindowAux::new()
                .x(self.panel_x)
                .y(self.panel_y),
        )?;
        self.conn.flush()?;
        Ok(())
    }

    pub fn poll(&mut self) -> Result<ChatEvent, Box<dyn std::error::Error>> {
        if !self.open {
            return Ok(ChatEvent::Closed);
        }

        while let Some(event) = self.conn.poll_for_event()? {
            match event {
                Event::KeyPress(e) => {
                    let result = self.handle_key(e)?;
                    match result {
                        ChatEvent::None => {}
                        other => return Ok(other),
                    }
                }
                Event::ButtonPress(e) => {
                    if e.detail == 4 {
                        self.scroll_up();
                        self.repaint()?;
                    } else if e.detail == 5 {
                        self.scroll_down();
                        self.repaint()?;
                    }
                }
                Event::FocusOut(_) => {
                    let _ = self.conn.set_input_focus(
                        InputFocus::PARENT,
                        self.window_id,
                        x11rb::CURRENT_TIME,
                    );
                    let _ = self.conn.flush();
                }
                Event::Expose(_) => {
                    self.repaint()?;
                }
                _ => {}
            }
        }

        Ok(ChatEvent::None)
    }

    pub fn close(&mut self) {
        self.open = false;
        let _ = self.conn.unmap_window(self.window_id);
        let _ = self.conn.flush();
    }

    fn handle_key(
        &mut self,
        event: KeyPressEvent,
    ) -> Result<ChatEvent, Box<dyn std::error::Error>> {
        let keycode = event.detail;

        match keycode {
            9 => {
                self.close();
                return Ok(ChatEvent::Closed);
            }
            36 | 104 => {
                if !self.input_text.is_empty() {
                    let text = self.input_text.clone();
                    self.input_text.clear();
                    self.messages.push(ChatMessage {
                        sender: "You".to_string(),
                        text: text.clone(),
                    });
                    self.scroll_to_bottom();
                    self.repaint()?;
                    return Ok(ChatEvent::MessageSent(text));
                }
            }
            22 => {
                self.input_text.pop();
                self.redraw_input()?;
            }
            _ => {
                if let Some(ch) = keycode_to_char(keycode, event.state) {
                    if self.input_text.len() < 300 {
                        self.input_text.push(ch);
                        self.redraw_input()?;
                    }
                }
            }
        }

        Ok(ChatEvent::None)
    }

    fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(3);
    }

    fn scroll_down(&mut self) {
        let total = self.rendered_line_count();
        let visible = visible_line_count();
        if total > visible {
            self.scroll_offset =
                (self.scroll_offset + 3).min(total - visible);
        }
    }

    fn scroll_to_bottom(&mut self) {
        let total = self.rendered_line_count();
        let visible = visible_line_count();
        if total > visible {
            self.scroll_offset = total - visible;
        } else {
            self.scroll_offset = 0;
        }
    }

    fn rendered_line_count(&self) -> usize {
        let mcpl = max_chars_per_line();
        let mut count = 0;
        for msg in &self.messages {
            count += 1;
            count += wrap_text(&msg.text, mcpl).len();
        }
        count
    }

    fn set_fg(&self, color: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.change_gc(
            self.gc,
            &ChangeGCAux::new().foreground(color),
        )?;
        Ok(())
    }

    fn fill_rect(
        &self,
        color: u32,
        x: i16,
        y: i16,
        w: u16,
        h: u16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.set_fg(color)?;
        self.conn.poly_fill_rectangle(
            self.window_id,
            self.gc,
            &[Rectangle { x, y, width: w, height: h }],
        )?;
        Ok(())
    }

    fn draw_text(
        &self,
        color: u32,
        x: i16,
        y: i16,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if text.is_empty() {
            return Ok(());
        }
        self.set_fg(color)?;
        self.conn.image_text8(
            self.window_id,
            self.gc,
            x,
            y,
            text.as_bytes(),
        )?;
        Ok(())
    }

    fn draw_hline(
        &self,
        color: u32,
        y: i16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.fill_rect(color, PADDING, y, PANEL_WIDTH - PADDING as u16 * 2, 1)
    }

    fn repaint(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Clear entire window
        self.fill_rect(BG, 0, 0, PANEL_WIDTH, PANEL_HEIGHT)?;

        // Border
        self.fill_rect(BORDER, 0, 0, PANEL_WIDTH, 1)?;
        self.fill_rect(BORDER, 0, PANEL_HEIGHT as i16 - 1, PANEL_WIDTH, 1)?;
        self.fill_rect(BORDER, 0, 0, 1, PANEL_HEIGHT)?;
        self.fill_rect(BORDER, PANEL_WIDTH as i16 - 1, 0, 1, PANEL_HEIGHT)?;

        // Title
        let title = "Chat with Lanty";
        let title_x = (PANEL_WIDTH as i16 - title.len() as i16 * CHAR_W) / 2;
        self.draw_text(TITLE_CLR, title_x, PADDING + CHAR_H - 2, title)?;

        // Header separator
        self.draw_hline(SEPARATOR, PADDING + HEADER_HEIGHT)?;

        // Chat messages
        self.draw_messages()?;

        // Input separator
        self.draw_hline(SEPARATOR, input_area_top() - 6)?;

        // Input area
        self.draw_input_area()?;

        self.conn.flush()?;
        Ok(())
    }

    fn redraw_input(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.draw_input_area()?;
        self.conn.flush()?;
        Ok(())
    }

    fn draw_messages(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mcpl = max_chars_per_line();
        let visible = visible_line_count();
        let top = chat_area_top();

        // Build all lines: (text, color)
        let mut all_lines: Vec<(String, u32)> = Vec::new();
        for msg in &self.messages {
            let (name_clr, text_clr) = match msg.sender.as_str() {
                "You" => (USER_NAME, USER_TEXT),
                "..." => (STATUS_CLR, STATUS_CLR),
                _ => (LANTY_NAME, LANTY_TEXT),
            };
            all_lines.push((format!("{}:", msg.sender), name_clr));
            for line in wrap_text(&msg.text, mcpl) {
                all_lines.push((format!("  {line}"), text_clr));
            }
        }

        let start = self.scroll_offset.min(all_lines.len());
        let end = (start + visible).min(all_lines.len());

        for (i, (line, color)) in all_lines[start..end].iter().enumerate() {
            let y = top + (i as i16) * LINE_HEIGHT + CHAR_H;
            self.draw_text(*color, PADDING, y, line)?;
        }

        Ok(())
    }

    fn draw_input_area(&self) -> Result<(), Box<dyn std::error::Error>> {
        let iat = input_area_top();
        let ibw = PANEL_WIDTH - PADDING as u16 * 2;
        let ibh = INPUT_HEIGHT as u16 - 4;

        // Clear input area
        self.fill_rect(BG, PADDING, iat - 1, ibw, ibh + 2)?;

        // Input box background
        self.fill_rect(INPUT_BG, PADDING, iat, ibw, ibh)?;

        // Input box border
        self.fill_rect(BORDER, PADDING, iat, ibw, 1)?;
        self.fill_rect(BORDER, PADDING, iat + ibh as i16 - 1, ibw, 1)?;
        self.fill_rect(BORDER, PADDING, iat, 1, ibh)?;
        self.fill_rect(BORDER, PADDING + ibw as i16 - 1, iat, 1, ibh)?;

        let text_y = iat + (ibh as i16 + CHAR_H) / 2 - 2;
        let text_x = PADDING + 8;

        if self.input_text.is_empty() {
            self.draw_text(PLACEHOLDER, text_x, text_y, "Type a message...")?;
        } else {
            let max_visible = ((ibw as i16 - 20) / CHAR_W) as usize;
            let t = &self.input_text;
            let visible = if t.len() > max_visible {
                &t[t.len() - max_visible..]
            } else {
                t.as_str()
            };
            self.draw_text(INPUT_TEXT, text_x, text_y, visible)?;

            // Cursor
            let cursor_x = text_x + visible.len() as i16 * CHAR_W + 2;
            let cursor_y = iat + (ibh as i16 - CHAR_H) / 2;
            self.fill_rect(CURSOR_CLR, cursor_x, cursor_y, 2, CHAR_H as u16)?;
        }

        Ok(())
    }
}

fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if current.is_empty() {
            current = word.to_string();
        } else if current.len() + 1 + word.len() <= max_chars {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(current);
            current = word.to_string();
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

fn keycode_to_char(keycode: u8, state: KeyButMask) -> Option<char> {
    let shift = state.contains(KeyButMask::SHIFT);

    // Standard US QWERTY keymap (evdev keycodes)
    let ch = match keycode {
        // Number row
        10 => if shift { '!' } else { '1' },
        11 => if shift { '@' } else { '2' },
        12 => if shift { '#' } else { '3' },
        13 => if shift { '$' } else { '4' },
        14 => if shift { '%' } else { '5' },
        15 => if shift { '^' } else { '6' },
        16 => if shift { '&' } else { '7' },
        17 => if shift { '*' } else { '8' },
        18 => if shift { '(' } else { '9' },
        19 => if shift { ')' } else { '0' },
        20 => if shift { '_' } else { '-' },
        21 => if shift { '+' } else { '=' },
        // QWERTY row
        24 => if shift { 'Q' } else { 'q' },
        25 => if shift { 'W' } else { 'w' },
        26 => if shift { 'E' } else { 'e' },
        27 => if shift { 'R' } else { 'r' },
        28 => if shift { 'T' } else { 't' },
        29 => if shift { 'Y' } else { 'y' },
        30 => if shift { 'U' } else { 'u' },
        31 => if shift { 'I' } else { 'i' },
        32 => if shift { 'O' } else { 'o' },
        33 => if shift { 'P' } else { 'p' },
        34 => if shift { '{' } else { '[' },
        35 => if shift { '}' } else { ']' },
        // ASDF row
        38 => if shift { 'A' } else { 'a' },
        39 => if shift { 'S' } else { 's' },
        40 => if shift { 'D' } else { 'd' },
        41 => if shift { 'F' } else { 'f' },
        42 => if shift { 'G' } else { 'g' },
        43 => if shift { 'H' } else { 'h' },
        44 => if shift { 'J' } else { 'j' },
        45 => if shift { 'K' } else { 'k' },
        46 => if shift { 'L' } else { 'l' },
        47 => if shift { ':' } else { ';' },
        48 => if shift { '"' } else { '\'' },
        49 => if shift { '~' } else { '`' },
        51 => if shift { '|' } else { '\\' },
        // ZXCV row
        52 => if shift { 'Z' } else { 'z' },
        53 => if shift { 'X' } else { 'x' },
        54 => if shift { 'C' } else { 'c' },
        55 => if shift { 'V' } else { 'v' },
        56 => if shift { 'B' } else { 'b' },
        57 => if shift { 'N' } else { 'n' },
        58 => if shift { 'M' } else { 'm' },
        59 => if shift { '<' } else { ',' },
        60 => if shift { '>' } else { '.' },
        61 => if shift { '?' } else { '/' },
        // Space
        65 => ' ',
        _ => return None,
    };
    Some(ch)
}
