use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MushState {
    Default,
    Happy,
    Angry,
    Bored,
    Carrying,
    Excited,
    Humming,
    Party,
    Proud,
    Raining,
    Scared,
    Sleepy,
    Sunglasses,
    Surprised,
    Talking,
    Thinking,
    Winking,
    Worried,
}

impl MushState {
    pub fn filename(&self) -> &'static str {
        match self {
            Self::Default => "mush-friend.svg",
            Self::Happy => "mush-happy.svg",
            Self::Angry => "mush-angry.svg",
            Self::Bored => "mush-bored.svg",
            Self::Carrying => "mush-carrying.svg",
            Self::Excited => "mush-excited.svg",
            Self::Humming => "mush-humming.svg",
            Self::Party => "mush-party.svg",
            Self::Proud => "mush-proud.svg",
            Self::Raining => "mush-raining.svg",
            Self::Scared => "mush-scared.svg",
            Self::Sleepy => "mush-sleepy.svg",
            Self::Sunglasses => "mush-sunglasses.svg",
            Self::Surprised => "mush-surprised.svg",
            Self::Talking => "mush-talking.svg",
            Self::Thinking => "mush-thinking.svg",
            Self::Winking => "mush-winking.svg",
            Self::Worried => "mush-worried.svg",
        }
    }

    pub const ALL: &[MushState] = &[
        Self::Default,
        Self::Happy,
        Self::Angry,
        Self::Bored,
        Self::Carrying,
        Self::Excited,
        Self::Humming,
        Self::Party,
        Self::Proud,
        Self::Raining,
        Self::Scared,
        Self::Sleepy,
        Self::Sunglasses,
        Self::Surprised,
        Self::Talking,
        Self::Thinking,
        Self::Winking,
        Self::Worried,
    ];

    pub const IDLE_STATES: &[MushState] = &[
        Self::Default,
        Self::Happy,
        Self::Bored,
        Self::Humming,
        Self::Thinking,
        Self::Winking,
        Self::Sunglasses,
        Self::Proud,
    ];
}

pub struct AppState {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub current_state: MushState,
    pub dragging: bool,
    pub drag_offset_x: i32,
    pub drag_offset_y: i32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub target_x: i32,
    pub target_y: i32,
    pub moving: bool,
    pub move_speed: f64,
    pub last_move_decision: Instant,
    pub move_decision_interval: Duration,
    pub last_state_change: Instant,
    pub idle_duration: Duration,
    pub needs_redraw: bool,
    pub last_comment: Instant,
    pub comment_interval: Duration,
    pub wants_comment: bool,
}

impl AppState {
    pub fn new(screen_width: u32, screen_height: u32, size: u32) -> Self {
        let aspect = 120.0 / 100.0;
        let w = size;
        let h = (size as f64 * aspect) as u32;
        let x = (screen_width / 2 - w / 2) as i32;
        let y = (screen_height - h - 60) as i32;

        Self {
            x,
            y,
            width: w,
            height: h,
            current_state: MushState::Default,
            dragging: false,
            drag_offset_x: 0,
            drag_offset_y: 0,
            screen_width,
            screen_height,
            target_x: x,
            target_y: y,
            moving: false,
            move_speed: 1.5,
            last_move_decision: Instant::now(),
            move_decision_interval: Duration::from_secs(5),
            last_state_change: Instant::now(),
            idle_duration: Duration::from_secs(8),
            needs_redraw: true,
            last_comment: Instant::now(),
            comment_interval: Duration::from_secs(300),
            wants_comment: false,
        }
    }
}
