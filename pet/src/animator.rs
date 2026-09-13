//! Lanty's behavior: drifts left/right, idles, breathes, swaps emotions.
//!
//! The animator owns world position, velocity, and the active `Emotion`.
//! Each frame the wayland loop calls `tick(dt)` and then reads `x()`, `y()`,
//! and `emotion()` to know what and where to draw.

use std::time::{Duration, Instant};

use crate::sprite::{Emotion, SPRITE_H, SPRITE_W};

/// Walking speed in surface pixels per second.
const WALK_SPEED: f32 = 35.0;
/// Floor offset from bottom of screen, in surface pixels.
const FLOOR_FROM_BOTTOM: f32 = 80.0;
/// How tall the breathing bob is, in surface pixels.
const BREATH_AMPLITUDE: f32 = 4.0;
/// Breathing cycle period in seconds.
const BREATH_PERIOD: f32 = 3.2;
/// Emotion rotation cadence.
const EMOTION_ROTATE_SECS: f32 = 12.0;
/// How long click reaction (Excited) lasts.
const REACTION_SECS: f32 = 2.5;

#[derive(Debug, Clone, Copy)]
enum Motion {
    Walking { dir: f32 }, // -1.0 or +1.0
    Idle,
}

pub struct Animator {
    pub screen_w: u32,
    pub screen_h: u32,
    x: f32,
    motion: Motion,
    motion_remaining: f32,
    elapsed: f32,
    emotion: Emotion,
    emotion_remaining: f32,
    reaction_remaining: f32,
    rng_state: u64,
    started: Instant,
}

impl Animator {
    pub fn new(screen_w: u32, screen_h: u32) -> Self {
        let mut a = Self {
            screen_w,
            screen_h,
            x: screen_w as f32 * 0.5 - SPRITE_W as f32 * 0.5,
            motion: Motion::Walking { dir: -1.0 },
            motion_remaining: 4.0,
            elapsed: 0.0,
            emotion: Emotion::Idle,
            emotion_remaining: EMOTION_ROTATE_SECS,
            reaction_remaining: 0.0,
            rng_state: 0x9E3779B97F4A7C15,
            started: Instant::now(),
        };
        // Seed RNG with startup time so each run feels different.
        let nanos = a.started.elapsed().as_nanos() as u64;
        a.rng_state ^= nanos.wrapping_mul(0xBF58476D1CE4E5B9);
        a
    }

    pub fn set_screen(&mut self, w: u32, h: u32) {
        self.screen_w = w;
        self.screen_h = h;
    }

    pub fn tick(&mut self, dt: Duration) {
        let dt_s = dt.as_secs_f32();
        self.elapsed += dt_s;

        // Motion update
        match self.motion {
            Motion::Walking { dir } => {
                self.x += dir * WALK_SPEED * dt_s;
                let max_x = (self.screen_w as f32 - SPRITE_W as f32).max(0.0);
                if self.x < 0.0 {
                    self.x = 0.0;
                    self.motion = Motion::Walking { dir: 1.0 };
                } else if self.x > max_x {
                    self.x = max_x;
                    self.motion = Motion::Walking { dir: -1.0 };
                }
            }
            Motion::Idle => {}
        }

        // Motion state transitions
        self.motion_remaining -= dt_s;
        if self.motion_remaining <= 0.0 {
            self.motion = match self.motion {
                Motion::Walking { .. } => {
                    self.motion_remaining = 2.5 + self.rand01() * 4.0;
                    Motion::Idle
                }
                Motion::Idle => {
                    self.motion_remaining = 4.0 + self.rand01() * 6.0;
                    Motion::Walking {
                        dir: if self.rand01() < 0.5 { -1.0 } else { 1.0 },
                    }
                }
            };
        }

        // Emotion rotation (paused during reaction)
        if self.reaction_remaining > 0.0 {
            self.reaction_remaining -= dt_s;
            if self.reaction_remaining <= 0.0 {
                self.emotion = self.pick_random_emotion();
                self.emotion_remaining = EMOTION_ROTATE_SECS;
            }
        } else {
            self.emotion_remaining -= dt_s;
            if self.emotion_remaining <= 0.0 {
                self.emotion = self.pick_random_emotion();
                self.emotion_remaining = EMOTION_ROTATE_SECS;
            }
        }
    }

    /// Trigger a click reaction: brief Excited emotion + pause walking.
    pub fn react_to_click(&mut self) {
        self.emotion = Emotion::Excited;
        self.reaction_remaining = REACTION_SECS;
        self.motion = Motion::Idle;
        self.motion_remaining = REACTION_SECS;
    }

    pub fn x(&self) -> i32 {
        self.x.round() as i32
    }

    /// Y is the *floor* minus the sprite height, plus the breathing bob.
    pub fn y(&self) -> i32 {
        let floor = self.screen_h as f32 - FLOOR_FROM_BOTTOM;
        let baseline = floor - SPRITE_H as f32;
        let bob = (self.elapsed * std::f32::consts::TAU / BREATH_PERIOD).sin()
            * BREATH_AMPLITUDE;
        (baseline + bob).round() as i32
    }

    pub fn emotion(&self) -> Emotion {
        self.emotion
    }

    pub fn bbox(&self) -> (i32, i32, u32, u32) {
        (self.x(), self.y(), SPRITE_W, SPRITE_H)
    }

    fn pick_random_emotion(&mut self) -> Emotion {
        const POOL: &[Emotion] = &[
            Emotion::Idle,
            Emotion::Happy,
            Emotion::Humming,
            Emotion::Thinking,
            Emotion::Sleepy,
        ];
        let idx = (self.rand01() * POOL.len() as f32) as usize;
        POOL[idx.min(POOL.len() - 1)]
    }

    fn rand01(&mut self) -> f32 {
        // xorshift64
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64 / u64::MAX as f64) as f32
    }
}
