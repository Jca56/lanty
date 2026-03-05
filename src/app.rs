use crate::brain::Brain;
use crate::bubble::SpeechBubble;
use crate::chat_input::{ChatPanel, ChatEvent};
use crate::personality::Personality;
use crate::renderer::SpriteCache;
use crate::state::{AppState, MushState};
use crate::window::DesktopWindow;
use rand::Rng;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use x11rb::protocol::Event;

const FRAME_DURATION: Duration = Duration::from_millis(33); // ~30fps

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let initial_size: u32 = 128;

    let assets_dir = find_assets_dir()?;
    println!("Assets directory: {}", assets_dir.display());

    let sprites = SpriteCache::load(&assets_dir, initial_size)?;

    let temp_win = DesktopWindow::new(1, 1, -10, -10)?;
    let (screen_w, screen_h) = temp_win.screen_size();
    drop(temp_win);

    let mut state = AppState::new(screen_w, screen_h, initial_size);
    let window = DesktopWindow::new(
        sprites.width(),
        sprites.height(),
        state.x,
        state.y,
    )?;

    window.paint(sprites.get_pixels(state.current_state))?;
    state.needs_redraw = false;

    let personality = Personality::new();
    let mut bubble = SpeechBubble::new();
    let mut brain = Brain::new();
    let mut chat_panel: Option<ChatPanel> = None;

    println!(
        "Lanty is alive! Screen: {}x{}, Position: ({}, {})",
        screen_w, screen_h, state.x, state.y,
    );

    // Show a greeting!
    bubble.show(
        "Hi! I'm Lanty! :D",
        state.x,
        state.y,
        state.width,
        Duration::from_secs(4),
    )?;

    let mut last_frame = Instant::now();

    loop {
        while let Some(event) = window.poll_event()? {
            handle_event(&event, &mut state, &mut chat_panel, screen_w, screen_h);
        }

        // Poll chat panel if active
        if let Some(ref mut panel) = chat_panel {
            match panel.poll()? {
                ChatEvent::MessageSent(text) => {
                    println!("User said: {text}");
                    brain.send_message(&text);
                    state.current_state = MushState::Thinking;
                    state.needs_redraw = true;
                    panel.add_status("Thinking...")?;
                }
                ChatEvent::Closed => {
                    chat_panel = None;
                }
                ChatEvent::None => {}
            }
        }

        // Poll brain for responses
        if let Some(result) = brain.poll_response() {
            match result {
                Ok(response) => {
                    println!("Lanty says: {response}");
                    state.current_state = MushState::Talking;
                    state.needs_redraw = true;
                    if let Some(ref mut panel) = chat_panel {
                        panel.remove_last_status()?;
                        panel.add_response(&response)?;
                    } else {
                        bubble.show(
                            &response,
                            state.x,
                            state.y,
                            state.width,
                            Duration::from_secs(8),
                        )?;
                    }
                }
                Err(e) => {
                    eprintln!("Brain error: {e}");
                    state.current_state = MushState::Worried;
                    state.needs_redraw = true;
                    if let Some(ref mut panel) = chat_panel {
                        panel.remove_last_status()?;
                        panel.add_response("My brain is foggy... is Ollama running?")?;
                    } else {
                        bubble.show(
                            "My brain is foggy... is Ollama running?",
                            state.x,
                            state.y,
                            state.width,
                            Duration::from_secs(5),
                        )?;
                    }
                }
            }
        }

        let now = Instant::now();
        if now - last_frame >= FRAME_DURATION {
            update_state(&mut state);

            if state.needs_redraw {
                window.paint(sprites.get_pixels(state.current_state))?;
                state.needs_redraw = false;
            }

            if state.dragging || state.moving {
                window.move_window(state.x, state.y)?;
            }

            // Speech bubble & chat panel: follow Lanty
            if state.dragging || state.moving {
                bubble.update_position(state.x, state.y, state.width)?;
                if let Some(ref mut panel) = chat_panel {
                    let _ = panel.reposition(
                        state.x, state.y, state.width,
                        screen_w, screen_h,
                    );
                }
            }

            if state.wants_comment {
                state.wants_comment = false;
                let comment = personality.random_comment();
                state.current_state = MushState::Talking;
                state.needs_redraw = true;
                bubble.show(
                    comment,
                    state.x,
                    state.y,
                    state.width,
                    Duration::from_secs(5),
                )?;
            }

            bubble.tick();
            if !bubble.is_visible()
                && state.current_state == MushState::Talking
            {
                state.current_state = MushState::Default;
                state.needs_redraw = true;
                state.last_state_change = Instant::now();
            }

            last_frame = now;
        }

        std::thread::sleep(Duration::from_millis(8));
    }
}

fn handle_event(
    event: &Event,
    state: &mut AppState,
    chat_panel: &mut Option<ChatPanel>,
    screen_w: u32,
    screen_h: u32,
) {
    match event {
        Event::ButtonPress(e) => {
            if e.detail == 1 {
                state.dragging = true;
                state.drag_offset_x = e.event_x as i32;
                state.drag_offset_y = e.event_y as i32;
                state.moving = false;
                state.current_state = MushState::Surprised;
                state.needs_redraw = true;
            } else if e.detail == 3 {
                // Right-click: toggle chat panel
                if let Some(ref mut panel) = chat_panel {
                    panel.close();
                    *chat_panel = None;
                } else {
                    match ChatPanel::open(
                        state.x, state.y, state.width,
                        screen_w, screen_h,
                    ) {
                        Ok(panel) => {
                            *chat_panel = Some(panel);
                            state.current_state = MushState::Excited;
                            state.needs_redraw = true;
                        }
                        Err(e) => eprintln!("Failed to open chat: {e}"),
                    }
                }
            }
        }
        Event::ButtonRelease(e) => {
            if e.detail == 1 && state.dragging {
                state.dragging = false;
                state.current_state = MushState::Happy;
                state.needs_redraw = true;
                state.last_state_change = Instant::now();
                // Resume wandering soon after being put down
                state.last_move_decision =
                    Instant::now() - state.move_decision_interval;
            }
        }
        Event::MotionNotify(e) => {
            if state.dragging {
                state.x = e.root_x as i32 - state.drag_offset_x;
                state.y = e.root_y as i32 - state.drag_offset_y;
                // Clamp to screen
                state.x = state.x.clamp(
                    0,
                    state.screen_width as i32 - state.width as i32,
                );
                state.y = state.y.clamp(
                    0,
                    state.screen_height as i32 - state.height as i32,
                );
            }
        }
        Event::Expose(_) => {
            state.needs_redraw = true;
        }
        _ => {}
    }
}

fn update_state(state: &mut AppState) {
    let now = Instant::now();
    let mut rng = rand::thread_rng();

    if state.dragging {
        return;
    }

    // Movement decisions
    if now - state.last_move_decision >= state.move_decision_interval {
        state.last_move_decision = now;

        if rng.gen_bool(0.55) {
            let margin = 60i32;
            state.target_x = rng.gen_range(
                margin..(state.screen_width as i32 - state.width as i32 - margin),
            );
            state.target_y = rng.gen_range(
                (state.screen_height as i32 / 3)
                    ..(state.screen_height as i32 - state.height as i32 - margin),
            );
            state.moving = true;
        } else {
            state.moving = false;
        }

        state.move_decision_interval =
            Duration::from_millis(rng.gen_range(3000..8000));
    }

    // Move toward target
    if state.moving {
        let dx = state.target_x - state.x;
        let dy = state.target_y - state.y;
        let dist = ((dx * dx + dy * dy) as f64).sqrt();

        if dist < state.move_speed * 2.0 {
            state.x = state.target_x;
            state.y = state.target_y;
            state.moving = false;
        } else {
            let nx = dx as f64 / dist;
            let ny = dy as f64 / dist;
            state.x += (nx * state.move_speed) as i32;
            state.y += (ny * state.move_speed) as i32;
        }
    }

    // Random mood changes during idle
    if !state.moving
        && now - state.last_state_change >= state.idle_duration
    {
        let idle_states = MushState::IDLE_STATES;
        let idx = rng.gen_range(0..idle_states.len());
        let new_state = idle_states[idx];
        if new_state != state.current_state {
            state.current_state = new_state;
            state.needs_redraw = true;
        }
        state.last_state_change = now;
        state.idle_duration =
            Duration::from_millis(rng.gen_range(4000..12000));
    }

    // Random comments
    if now - state.last_comment >= state.comment_interval {
        state.last_comment = now;
        state.wants_comment = true;
        state.comment_interval =
            Duration::from_millis(rng.gen_range(300_000..600_000));
    }
}

fn find_assets_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let exe_dir = std::env::current_exe()?
        .parent()
        .ok_or("No parent directory for executable")?
        .to_path_buf();

    let candidates = [
        exe_dir.join("assets"),
        exe_dir.join("../assets"),
        exe_dir.join("../../assets"),
        PathBuf::from("assets"),
    ];

    for path in &candidates {
        if path.join("mushrooms").exists() {
            return Ok(path.canonicalize()?);
        }
    }

    Err("Could not find assets directory. Make sure 'assets/mushrooms/' exists.".into())
}
