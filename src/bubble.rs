use crate::window::DesktopWindow;
use std::sync::Arc;
use std::time::{Duration, Instant};

const BUBBLE_PADDING: u32 = 16;
const CHARS_PER_LINE: usize = 24;
const LINE_HEIGHT: u32 = 20;
const FONT_SIZE: u32 = 18;
const TAIL_HEIGHT: u32 = 14;
const BUBBLE_MIN_WIDTH: u32 = 100;
const BUBBLE_MAX_WIDTH: u32 = 320;

pub struct SpeechBubble {
    window: Option<DesktopWindow>,
    fontdb: Arc<usvg::fontdb::Database>,
    visible: bool,
    show_until: Instant,
    current_text: String,
    bubble_width: u32,
    bubble_height: u32,
}

impl SpeechBubble {
    pub fn new() -> Self {
        let mut fontdb = usvg::fontdb::Database::new();
        fontdb.load_system_fonts();
        Self {
            window: None,
            fontdb: Arc::new(fontdb),
            visible: false,
            show_until: Instant::now(),
            current_text: String::new(),
            bubble_width: 0,
            bubble_height: 0,
        }
    }

    pub fn show(
        &mut self,
        text: &str,
        lanty_x: i32,
        lanty_y: i32,
        lanty_width: u32,
        duration: Duration,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let lines = wrap_text(text, CHARS_PER_LINE);
        let num_lines = lines.len() as u32;

        let max_line_len = lines.iter().map(|l| l.len()).max().unwrap_or(1);
        let text_width = (max_line_len as u32) * (FONT_SIZE * 6 / 10);
        let bw = (text_width + BUBBLE_PADDING * 2).clamp(BUBBLE_MIN_WIDTH, BUBBLE_MAX_WIDTH);
        let bh = num_lines * LINE_HEIGHT + BUBBLE_PADDING * 2 + TAIL_HEIGHT;

        let x = lanty_x + (lanty_width as i32 / 2) - (bw as i32 / 2);
        let y = lanty_y - bh as i32 - 4;

        let svg = build_bubble_svg(bw, bh, &lines);
        let pixels = render_bubble_svg(&svg, bw, bh, &self.fontdb)?;

        // Destroy old window and create new one at correct size/position
        self.window = None;
        let win = DesktopWindow::new(bw, bh, x.max(0), y.max(0))?;
        win.paint(&pixels)?;
        self.window = Some(win);

        self.visible = true;
        self.show_until = Instant::now() + duration;
        self.current_text = text.to_string();
        self.bubble_width = bw;
        self.bubble_height = bh;

        Ok(())
    }

    pub fn update_position(
        &self,
        lanty_x: i32,
        lanty_y: i32,
        lanty_width: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref win) = self.window {
            if self.visible {
                let x = lanty_x + (lanty_width as i32 / 2) - (self.bubble_width as i32 / 2);
                let y = lanty_y - self.bubble_height as i32 - 4;
                win.move_window(x.max(0), y.max(0))?;
            }
        }
        Ok(())
    }

    pub fn tick(&mut self) {
        if self.visible && Instant::now() >= self.show_until {
            self.visible = false;
            self.window = None;
        }
    }

    pub fn is_visible(&self) -> bool {
        self.visible
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

fn build_bubble_svg(width: u32, height: u32, lines: &[String]) -> String {
    let body_h = height - TAIL_HEIGHT;
    let radius = 12;
    let tail_cx = width / 2;

    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <filter id="bs">
      <feDropShadow dx="1" dy="2" stdDeviation="3" flood-color="#000" flood-opacity="0.25"/>
    </filter>
  </defs>
  <g filter="url(#bs)">
    <rect x="2" y="2" width="{bw}" height="{bh}" rx="{radius}" ry="{radius}"
          fill="#fff8f0" stroke="#c8b89a" stroke-width="1.5"/>
    <polygon points="{x1},{bt} {x2},{bt} {x3},{tt}"
             fill="#fff8f0" stroke="#c8b89a" stroke-width="1.5"/>
    <rect x="3" y="{cover_y}" width="{cover_w}" height="8"
          fill="#fff8f0"/>
  </g>"##,
        bw = width - 4,
        bh = body_h - 4,
        bt = body_h - 4,
        tt = height - 4,
        x1 = tail_cx - 10,
        x2 = tail_cx + 10,
        x3 = tail_cx,
        cover_y = body_h - 7,
        cover_w = 22,
    );

    // Fix cover rect position to center over tail
    let cover_x = tail_cx - 11;
    svg = svg.replace(
        &format!("x=\"3\" y=\"{}\"", body_h - 7),
        &format!("x=\"{}\" y=\"{}\"", cover_x, body_h - 7),
    );

    for (i, line) in lines.iter().enumerate() {
        let ty = BUBBLE_PADDING + 2 + (i as u32 + 1) * LINE_HEIGHT - 4;
        // Escape XML special characters
        let escaped = line
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;");
        svg.push_str(&format!(
            r##"
  <text x="{}" y="{}" font-family="sans-serif" font-size="{}" fill="#3a2a1a" text-anchor="middle">{}</text>"##,
            width / 2,
            ty,
            FONT_SIZE,
            escaped,
        ));
    }

    svg.push_str("\n</svg>");
    svg
}

fn render_bubble_svg(
    svg_data: &str,
    width: u32,
    height: u32,
    fontdb: &Arc<usvg::fontdb::Database>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let options = usvg::Options {
        fontdb: Arc::clone(fontdb),
        ..Default::default()
    };
    let tree = usvg::Tree::from_str(svg_data, &options)?;

    let mut pixmap =
        tiny_skia::Pixmap::new(width, height).ok_or("Failed to create pixmap")?;

    let svg_size = tree.size();
    let sx = width as f32 / svg_size.width();
    let sy = height as f32 / svg_size.height();
    let transform = tiny_skia::Transform::from_scale(sx, sy);

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    let rgba = pixmap.data();
    let mut bgra = vec![0u8; rgba.len()];
    for i in (0..rgba.len()).step_by(4) {
        bgra[i] = rgba[i + 2];
        bgra[i + 1] = rgba[i + 1];
        bgra[i + 2] = rgba[i];
        bgra[i + 3] = rgba[i + 3];
    }

    Ok(bgra)
}
