//! Wayland layer-shell client: registers globals, creates a fullscreen
//! transparent overlay surface, drives the frame loop, and routes pointer
//! clicks. Everything other than animation and sprite work lives here.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use wayland_client::protocol::{
    wl_buffer::{self, WlBuffer},
    wl_callback::{self, WlCallback},
    wl_compositor::WlCompositor,
    wl_output::{self, WlOutput},
    wl_pointer::{self, WlPointer},
    wl_region::WlRegion,
    wl_registry::{self, WlRegistry},
    wl_seat::{self, Capability, WlSeat},
    wl_shm::WlShm,
    wl_shm_pool::WlShmPool,
    wl_surface::WlSurface,
};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, WEnum};
use wayland_protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::{Layer, ZwlrLayerShellV1},
    zwlr_layer_surface_v1::{self, Anchor, KeyboardInteractivity, ZwlrLayerSurfaceV1},
};

use crate::animator::Animator;
use crate::shm::ShmPool;
use crate::sprite::{blit_premul_rgba_to_argb8888, SpriteSet};

const BTN_LEFT: u32 = 0x110;

pub struct App {
    pub icons_dir: PathBuf,
    pub launch_cmd: PathBuf,
}

impl App {
    pub fn run(self) -> Result<()> {
        let conn = Connection::connect_to_env().context("connect to wayland")?;
        let display = conn.display();
        let mut event_queue = conn.new_event_queue::<State>();
        let qh = event_queue.handle();
        let _registry = display.get_registry(&qh, ());

        let sprites = SpriteSet::load(&self.icons_dir).context("load sprites")?;

        let mut state = State {
            qh: qh.clone(),
            compositor: None,
            shm: None,
            seat: None,
            output: None,
            layer_shell: None,
            surface: None,
            layer_surface: None,
            pointer: None,
            sprites,
            animator: Animator::new(800, 600),
            pool: None,
            buf_index: 0,
            configured: false,
            screen_w: 0,
            screen_h: 0,
            pointer_x: -1.0,
            pointer_y: -1.0,
            pointer_pressed: false,
            launch_cmd: self.launch_cmd,
            last_tick: Instant::now(),
            exit: false,
        };

        // Roundtrip to receive all globals.
        event_queue.roundtrip(&mut state)?;
        tracing::info!(
            "globals: compositor={} shm={} seat={} output={} layer_shell={}",
            state.compositor.is_some(),
            state.shm.is_some(),
            state.seat.is_some(),
            state.output.is_some(),
            state.layer_shell.is_some(),
        );
        state.bind_required(&qh)?;
        // Second roundtrip in case seat caps arrive after.
        event_queue.roundtrip(&mut state)?;
        tracing::info!("layer surface created, awaiting configure");

        // Configure layer surface. Anchor to all four sides so the compositor
        // sizes us to the full output; exclusive_zone=-1 so we don't push
        // other clients; no keyboard input.
        let layer_surface = state.layer_surface.as_ref().unwrap();
        layer_surface.set_anchor(
            Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right,
        );
        layer_surface.set_exclusive_zone(-1);
        layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer_surface.set_size(0, 0);
        state.surface.as_ref().unwrap().commit();

        // Drive event loop until the compositor closes us.
        while !state.exit {
            event_queue.blocking_dispatch(&mut state)?;
        }
        Ok(())
    }
}

pub struct State {
    qh: QueueHandle<Self>,
    compositor: Option<WlCompositor>,
    shm: Option<WlShm>,
    seat: Option<WlSeat>,
    output: Option<WlOutput>,
    layer_shell: Option<ZwlrLayerShellV1>,
    surface: Option<WlSurface>,
    layer_surface: Option<ZwlrLayerSurfaceV1>,
    pointer: Option<WlPointer>,

    sprites: SpriteSet,
    animator: Animator,
    pool: Option<ShmPool>,
    buf_index: usize,
    configured: bool,
    screen_w: u32,
    screen_h: u32,

    pointer_x: f64,
    pointer_y: f64,
    pointer_pressed: bool,

    launch_cmd: PathBuf,
    last_tick: Instant,
    exit: bool,
}

impl State {
    fn bind_required(&mut self, qh: &QueueHandle<Self>) -> Result<()> {
        let compositor = self
            .compositor
            .as_ref()
            .ok_or_else(|| anyhow!("no wl_compositor"))?;
        let layer_shell = self
            .layer_shell
            .as_ref()
            .ok_or_else(|| anyhow!("compositor lacks zwlr_layer_shell_v1"))?;

        let surface = compositor.create_surface(qh, ());
        let layer_surface = layer_shell.get_layer_surface(
            &surface,
            self.output.as_ref(),
            Layer::Overlay,
            "lanty-pet".into(),
            qh,
            (),
        );
        self.surface = Some(surface);
        self.layer_surface = Some(layer_surface);
        Ok(())
    }

    fn ensure_pool(&mut self) -> Result<()> {
        if self.pool.is_some() {
            return Ok(());
        }
        let shm = self
            .shm
            .as_ref()
            .ok_or_else(|| anyhow!("no wl_shm"))?
            .clone();
        let pool = ShmPool::create(&shm, &self.qh, self.screen_w, self.screen_h)?;
        self.pool = Some(pool);
        Ok(())
    }

    fn render_frame(&mut self) -> Result<()> {
        if !self.configured {
            return Ok(());
        }
        if self.pool.is_none() {
            tracing::info!("first render at {}x{}", self.screen_w, self.screen_h);
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick);
        self.last_tick = now;
        self.animator.tick(dt);

        self.ensure_pool()?;
        let pool = self.pool.as_mut().unwrap();
        let idx = self.buf_index;
        let stride = pool.stride;
        let screen_w = pool.width;
        let screen_h = pool.height;
        let region = pool.region_mut(idx);
        // Clear to fully transparent.
        region.fill(0);

        let pixmap = self.sprites.get(self.animator.emotion());
        blit_premul_rgba_to_argb8888(
            pixmap,
            region,
            screen_w,
            screen_h,
            stride,
            self.animator.x(),
            self.animator.y(),
        );

        let surface = self.surface.as_ref().unwrap().clone();
        let compositor = self.compositor.as_ref().unwrap().clone();
        let buffer = pool.buffers[idx].clone();

        // Limit input to the sprite's bbox so clicks elsewhere pass through.
        let (bx, by, bw, bh) = self.animator.bbox();
        let region_obj = compositor.create_region(&self.qh, ());
        region_obj.add(bx, by, bw as i32, bh as i32);
        surface.set_input_region(Some(&region_obj));
        region_obj.destroy();

        surface.attach(Some(&buffer), 0, 0);
        surface.damage_buffer(0, 0, screen_w as i32, screen_h as i32);
        surface.frame(&self.qh, ());
        surface.commit();

        self.buf_index = 1 - self.buf_index;
        Ok(())
    }

    fn handle_click(&mut self) {
        let (bx, by, bw, bh) = self.animator.bbox();
        let x = self.pointer_x as i32;
        let y = self.pointer_y as i32;
        if x < bx || x >= bx + bw as i32 || y < by || y >= by + bh as i32 {
            return;
        }
        self.animator.react_to_click();
        match Command::new(&self.launch_cmd).spawn() {
            Ok(_) => tracing::info!("launched chat: {}", self.launch_cmd.display()),
            Err(e) => tracing::warn!("failed to launch chat: {}", e),
        }
    }
}

// -------------------------------------------------------------- registry

impl Dispatch<WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(registry.bind::<WlCompositor, _, _>(
                        name,
                        version.min(4),
                        qh,
                        (),
                    ));
                }
                "wl_shm" => {
                    state.shm = Some(registry.bind::<WlShm, _, _>(name, 1, qh, ()));
                }
                "wl_seat" => {
                    state.seat = Some(registry.bind::<WlSeat, _, _>(
                        name,
                        version.min(7),
                        qh,
                        (),
                    ));
                }
                "wl_output" if state.output.is_none() => {
                    state.output = Some(registry.bind::<WlOutput, _, _>(
                        name,
                        version.min(4),
                        qh,
                        (),
                    ));
                }
                "zwlr_layer_shell_v1" => {
                    state.layer_shell = Some(registry.bind::<ZwlrLayerShellV1, _, _>(
                        name,
                        version.min(4),
                        qh,
                        (),
                    ));
                }
                _ => {}
            }
        }
    }
}

// ------------------------------------------------------------ layer surface

impl Dispatch<ZwlrLayerSurfaceV1, ()> for State {
    fn event(
        state: &mut Self,
        layer: &ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_layer_surface_v1::Event::Configure { serial, width, height } => {
                tracing::info!("layer configure: {}x{}", width, height);
                layer.ack_configure(serial);
                if width > 0 && height > 0 {
                    state.screen_w = width;
                    state.screen_h = height;
                }
                state.animator.set_screen(state.screen_w, state.screen_h);
                state.configured = true;
                // Pool's size is fixed at first configure; if the compositor
                // resizes us later we'd need to reallocate.
                if let Err(e) = state.render_frame() {
                    tracing::error!("render: {e:?}");
                }
            }
            zwlr_layer_surface_v1::Event::Closed => {
                state.exit = true;
            }
            _ => {}
        }
    }
}

// ----------------------------------------------------------- frame callback

impl Dispatch<WlCallback, ()> for State {
    fn event(
        state: &mut Self,
        _: &WlCallback,
        event: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            if let Err(e) = state.render_frame() {
                tracing::error!("render: {e:?}");
            }
        }
    }
}

// ----------------------------------------------------------------- seat

impl Dispatch<WlSeat, ()> for State {
    fn event(
        state: &mut Self,
        seat: &WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities { capabilities } = event {
            let caps = match capabilities {
                WEnum::Value(c) => c,
                WEnum::Unknown(_) => return,
            };
            if caps.contains(Capability::Pointer) && state.pointer.is_none() {
                state.pointer = Some(seat.get_pointer(qh, ()));
            }
        }
    }
}

// ---------------------------------------------------------------- pointer

impl Dispatch<WlPointer, ()> for State {
    fn event(
        state: &mut Self,
        _: &WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter { surface_x, surface_y, .. } => {
                state.pointer_x = surface_x;
                state.pointer_y = surface_y;
            }
            wl_pointer::Event::Motion { surface_x, surface_y, .. } => {
                state.pointer_x = surface_x;
                state.pointer_y = surface_y;
            }
            wl_pointer::Event::Leave { .. } => {
                state.pointer_x = -1.0;
                state.pointer_y = -1.0;
            }
            wl_pointer::Event::Button { button, state: btn_state, .. } => {
                if button != BTN_LEFT {
                    return;
                }
                let pressed = matches!(
                    btn_state,
                    WEnum::Value(wl_pointer::ButtonState::Pressed)
                );
                if pressed && !state.pointer_pressed {
                    state.handle_click();
                }
                state.pointer_pressed = pressed;
            }
            _ => {}
        }
    }
}

// ------------------------------------------------------------ noop dispatch

macro_rules! noop_dispatch {
    ($($t:ty),+ $(,)?) => {
        $(impl Dispatch<$t, ()> for State {
            fn event(
                _: &mut Self,
                _: &$t,
                _: <$t as Proxy>::Event,
                _: &(),
                _: &Connection,
                _: &QueueHandle<Self>,
            ) {}
        })+
    };
}

noop_dispatch!(
    WlCompositor,
    WlShm,
    WlShmPool,
    WlSurface,
    WlRegion,
    ZwlrLayerShellV1,
);

impl Dispatch<WlBuffer, ()> for State {
    fn event(
        _: &mut Self,
        _: &WlBuffer,
        event: wl_buffer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // We could track release here for safer multi-buffering; with two
        // buffers and frame-callback-driven rendering, by the time we want
        // to write a buffer the compositor has released it.
        let _ = event;
    }
}

impl Dispatch<WlOutput, ()> for State {
    fn event(
        _: &mut Self,
        _: &WlOutput,
        event: wl_output::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // We let the compositor size us via layer_surface configure; output
        // info is informational only here.
        let _ = event;
    }
}
