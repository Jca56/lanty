//! SVG emotion sprites rasterized at startup into tiny-skia pixmaps.
//!
//! Each `Emotion` maps to a file in the Lanty icon set; we render each one
//! once at the target sprite size and keep the pixmap in memory. The pixmap
//! is RGBA premultiplied; the wayland shm copy path is responsible for any
//! channel reordering.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use tiny_skia::{Pixmap, Transform};

/// Logical sprite width in surface coordinates.
pub const SPRITE_W: u32 = 160;
/// Logical sprite height. Lanty SVGs are 100:120 (5:6) — match that aspect.
pub const SPRITE_H: u32 = 192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Emotion {
    Idle,
    Happy,
    Thinking,
    Humming,
    Excited,
    Sleepy,
    Surprised,
}

impl Emotion {
    pub fn filename(self) -> &'static str {
        match self {
            Emotion::Idle => "lanty.svg",
            Emotion::Happy => "mush-happy.svg",
            Emotion::Thinking => "mush-thinking.svg",
            Emotion::Humming => "mush-humming.svg",
            Emotion::Excited => "mush-excited.svg",
            Emotion::Sleepy => "mush-sleepy.svg",
            Emotion::Surprised => "mush-surprised.svg",
        }
    }

    pub fn all() -> &'static [Emotion] {
        &[
            Emotion::Idle,
            Emotion::Happy,
            Emotion::Thinking,
            Emotion::Humming,
            Emotion::Excited,
            Emotion::Sleepy,
            Emotion::Surprised,
        ]
    }
}

pub struct SpriteSet {
    pixmaps: HashMap<Emotion, Pixmap>,
}

impl SpriteSet {
    pub fn load(icons_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = icons_dir.as_ref();
        let mut pixmaps = HashMap::new();
        for emotion in Emotion::all() {
            let path: PathBuf = dir.join(emotion.filename());
            let pixmap = rasterize(&path, SPRITE_W, SPRITE_H).with_context(|| {
                format!("rasterize {:?} from {}", emotion, path.display())
            })?;
            pixmaps.insert(*emotion, pixmap);
        }
        Ok(Self { pixmaps })
    }

    pub fn get(&self, emotion: Emotion) -> &Pixmap {
        self.pixmaps
            .get(&emotion)
            .unwrap_or_else(|| self.pixmaps.get(&Emotion::Idle).expect("idle loaded"))
    }
}

fn rasterize(path: &Path, width: u32, height: u32) -> Result<Pixmap> {
    let data = std::fs::read(path)
        .with_context(|| format!("read SVG {}", path.display()))?;
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_data(&data, &opt)
        .with_context(|| format!("parse SVG {}", path.display()))?;

    let svg_size = tree.size();
    let sx = width as f32 / svg_size.width();
    let sy = height as f32 / svg_size.height();
    let scale = sx.min(sy); // preserve aspect

    let scaled_w = (svg_size.width() * scale).ceil() as u32;
    let scaled_h = (svg_size.height() * scale).ceil() as u32;
    let ox = ((width as i32 - scaled_w as i32) / 2) as f32;
    let oy = ((height as i32 - scaled_h as i32) / 2) as f32;

    let mut pixmap = Pixmap::new(width, height)
        .ok_or_else(|| anyhow!("alloc pixmap {}x{}", width, height))?;
    let transform = Transform::from_scale(scale, scale).post_translate(ox, oy);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    Ok(pixmap)
}

/// Copy a `tiny-skia` pixmap (RGBA premultiplied) into a wl_shm `Argb8888`
/// region (little-endian = memory order BGRA premultiplied).
///
/// `dst` is the full surface buffer, `dst_stride` is its byte stride.
/// `(dx, dy)` is the top-left blit position in pixels. Clipped to the
/// destination bounds.
pub fn blit_premul_rgba_to_argb8888(
    src: &Pixmap,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    dst_stride: usize,
    dx: i32,
    dy: i32,
) {
    let sw = src.width() as i32;
    let sh = src.height() as i32;
    let src_bytes = src.data();

    let x0 = dx.max(0);
    let y0 = dy.max(0);
    let x1 = (dx + sw).min(dst_w as i32);
    let y1 = (dy + sh).min(dst_h as i32);
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    for y in y0..y1 {
        let sy = (y - dy) as usize;
        let dst_row = y as usize * dst_stride;
        for x in x0..x1 {
            let sx = (x - dx) as usize;
            let si = (sy * sw as usize + sx) * 4;
            let di = dst_row + x as usize * 4;
            let r = src_bytes[si];
            let g = src_bytes[si + 1];
            let b = src_bytes[si + 2];
            let a = src_bytes[si + 3];
            // Argb8888 little-endian memory: B, G, R, A
            dst[di] = b;
            dst[di + 1] = g;
            dst[di + 2] = r;
            dst[di + 3] = a;
        }
    }
}
