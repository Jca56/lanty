use x11rb::connection::{Connection, RequestConnection};
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

pub struct DesktopWindow {
    conn: RustConnection,
    screen_num: usize,
    window_id: Window,
    gc: Gcontext,
    width: u32,
    height: u32,
    depth: u8,
}

impl DesktopWindow {
    pub fn new(
        width: u32,
        height: u32,
        x: i32,
        y: i32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (conn, screen_num) = x11rb::connect(None)?;
        let screen = &conn.setup().roots[screen_num];

        let (depth, visual_id) = find_argb_visual(screen)
            .ok_or("No 32-bit ARGB visual found. Is a compositor running?")?;

        let colormap = conn.generate_id()?;
        conn.create_colormap(ColormapAlloc::NONE, colormap, screen.root, visual_id)?;

        let window_id = conn.generate_id()?;
        let values = CreateWindowAux::new()
            .background_pixel(0)
            .border_pixel(0)
            .override_redirect(1)
            .colormap(colormap)
            .event_mask(
                EventMask::EXPOSURE
                    | EventMask::BUTTON_PRESS
                    | EventMask::BUTTON_RELEASE
                    | EventMask::POINTER_MOTION
                    | EventMask::STRUCTURE_NOTIFY,
            );

        conn.create_window(
            depth,
            window_id,
            screen.root,
            x as i16,
            y as i16,
            width as u16,
            height as u16,
            0,
            WindowClass::INPUT_OUTPUT,
            visual_id,
            &values,
        )?;

        set_wm_hints(&conn, window_id)?;

        let gc = conn.generate_id()?;
        conn.create_gc(gc, window_id, &CreateGCAux::new())?;

        conn.map_window(window_id)?;
        conn.flush()?;

        Ok(Self {
            conn,
            screen_num,
            window_id,
            gc,
            width,
            height,
            depth,
        })
    }

    pub fn paint(&self, pixels: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // X11 put_image has a max request size. For large images, we send in strips.
        let bytes_per_row = (self.width * 4) as usize;
        let max_request = self.conn.maximum_request_bytes();
        // Leave room for the request header (~28 bytes)
        let max_rows = (max_request - 64) / bytes_per_row;
        let max_rows = max_rows.max(1);

        let mut y_offset: u16 = 0;
        let mut remaining = self.height as usize;

        while remaining > 0 {
            let rows = remaining.min(max_rows);
            let start = y_offset as usize * bytes_per_row;
            let end = start + rows * bytes_per_row;

            self.conn.put_image(
                ImageFormat::Z_PIXMAP,
                self.window_id,
                self.gc,
                self.width as u16,
                rows as u16,
                0,
                y_offset as i16,
                0,
                self.depth,
                &pixels[start..end],
            )?;

            y_offset += rows as u16;
            remaining -= rows;
        }

        self.conn.flush()?;
        Ok(())
    }

    pub fn move_window(&self, x: i32, y: i32) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.configure_window(
            self.window_id,
            &ConfigureWindowAux::new().x(x).y(y),
        )?;
        self.conn.flush()?;
        Ok(())
    }

    pub fn poll_event(&self) -> Result<Option<Event>, Box<dyn std::error::Error>> {
        Ok(self.conn.poll_for_event()?)
    }

    pub fn screen_size(&self) -> (u32, u32) {
        let screen = &self.conn.setup().roots[self.screen_num];
        (
            screen.width_in_pixels as u32,
            screen.height_in_pixels as u32,
        )
    }

    pub fn raise(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.configure_window(
            self.window_id,
            &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE),
        )?;
        self.conn.flush()?;
        Ok(())
    }
}

fn find_argb_visual(screen: &Screen) -> Option<(u8, Visualid)> {
    for depth_info in &screen.allowed_depths {
        if depth_info.depth == 32 {
            for visual in &depth_info.visuals {
                if visual.class == VisualClass::TRUE_COLOR {
                    return Some((32, visual.visual_id));
                }
            }
        }
    }
    None
}

fn set_wm_hints(
    conn: &RustConnection,
    window: Window,
) -> Result<(), Box<dyn std::error::Error>> {
    // Set WM_NAME
    conn.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"Lanty",
    )?;

    // Set _NET_WM_NAME
    let net_wm_name = conn.intern_atom(false, b"_NET_WM_NAME")?.reply()?.atom;
    let utf8_string = conn.intern_atom(false, b"UTF8_STRING")?.reply()?.atom;
    conn.change_property8(
        PropMode::REPLACE,
        window,
        net_wm_name,
        utf8_string,
        b"Lanty",
    )?;

    Ok(())
}
