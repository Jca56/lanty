use crate::state::MushState;
use std::collections::HashMap;
use std::path::Path;

pub struct SpriteCache {
    sprites: HashMap<MushState, Vec<u8>>,
    width: u32,
    height: u32,
}

impl SpriteCache {
    pub fn load(assets_dir: &Path, size: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let aspect = 120.0 / 100.0;
        let width = size;
        let height = (size as f64 * aspect) as u32;
        let mut sprites = HashMap::new();

        let mushrooms_dir = assets_dir.join("mushrooms");

        for state in MushState::ALL {
            let path = mushrooms_dir.join(state.filename());
            let svg_data = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
            let pixels = render_svg_to_bgra(&svg_data, width, height)?;
            sprites.insert(*state, pixels);
        }

        println!(
            "Loaded {} sprites at {}x{}",
            sprites.len(),
            width,
            height,
        );

        Ok(Self {
            sprites,
            width,
            height,
        })
    }

    pub fn get_pixels(&self, state: MushState) -> &[u8] {
        self.sprites.get(&state).expect("Missing sprite state")
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

fn render_svg_to_bgra(
    svg_data: &str,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let options = usvg::Options::default();
    let tree = usvg::Tree::from_str(svg_data, &options)?;

    let mut pixmap =
        tiny_skia::Pixmap::new(width, height).ok_or("Failed to create pixmap")?;

    let svg_size = tree.size();
    let sx = width as f32 / svg_size.width();
    let sy = height as f32 / svg_size.height();
    let transform = tiny_skia::Transform::from_scale(sx, sy);

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    // tiny-skia gives premultiplied RGBA. X11 on little-endian expects BGRA.
    let rgba = pixmap.data();
    let mut bgra = vec![0u8; rgba.len()];
    for i in (0..rgba.len()).step_by(4) {
        bgra[i] = rgba[i + 2];     // B
        bgra[i + 1] = rgba[i + 1]; // G
        bgra[i + 2] = rgba[i];     // R
        bgra[i + 3] = rgba[i + 3]; // A
    }

    Ok(bgra)
}
