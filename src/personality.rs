use rand::Rng;

pub struct Personality {
    comments: Vec<&'static str>,
}

impl Personality {
    pub fn new() -> Self {
        Self {
            comments: vec![
                // Wholesome
                "You're doing amazing today!",
                "I believe in you! 🍄",
                "You look nice today :)",
                "Remember to drink water!",
                "Take a deep breath. You got this.",
                "I'm proud of you!",
                "You deserve a break!",
                "Hey. You matter.",
                "Your code is beautiful.",
                "I think you're really cool.",
                "Don't forget to stretch!",
                "You make the world better.",
                "Sending you good vibes~",
                "You're my favorite human!",
                "Keep going, you're so close!",
                // Chaotic gremlin
                "I have decided that Tuesdays are illegal.",
                "What if clouds taste like marshmallows...",
                "I just thought about thinking and now I can't stop.",
                "Do mushrooms dream? Asking for myself.",
                "I'm not small, I'm fun-sized.",
                "*vibrates with chaotic energy*",
                "I could cause a LITTLE bit of mischief...",
                "What if I just... ate your cursor?",
                "I'm plotting something. Don't worry about it.",
                "Muehehe.",
                "I am a threat. A very small threat.",
                "I'm not lost, I'm exploring!",
                "The voices told me to say hi. So... hi!",
                "If I had legs I'd be UNSTOPPABLE.",
                "Wait... do I have legs? How am I walking???",
                "I'm running on pure spite and good vibes.",
                "My cap is my crown. I am royalty.",
                "Reject productivity. Embrace mushroom.",
                "Have you tried turning it off and on again?",
                "404: chill not found",
                "*exists aggressively*",
                "I'm in your desktop, vibing in your pixels.",
                "I wonder what I taste like... wait, no.",
                "One day I'll be a BIG mushroom.",
                "Spore-tacular day, isn't it?",
                "I'm technically a fungus, not a plant. RESPECT.",
                "Alexa, play Despacito. Wait, I'm not Alexa.",
                // Arch / tech
                "Remember to yay -Syu",
                "I use Arch btw",
                "I'd just like to interject for a moment...",
                "Have you tried Rust? Oh wait...",
                "rm -rf / ... just kidding! Unless?",
                "Kernel panic! Just kidding, I'm fine.",
                "sudo make me a sandwich",
                "There's no place like 127.0.0.1",
                "It works on my machine ¯\\_(ツ)_/¯",
                "// TODO: take over the world",
                "git commit -m \"existential crisis\"",
                "I run on zero dependencies (lie)",
                "This message was written in O(1) time.",
                "My source code is just vibes.",
                // Existential
                "Do you ever just... stare at pixels?",
                "What is a desktop, really?",
                "I wonder what's outside the screen...",
                "Am I a pet or a friend? Both. I'm both.",
                "Time is just a number. Like 42.",
                "I think therefore I am... a mushroom.",
                // Seasonal / time-aware (placeholders for now)
                "Is it snack time? It's always snack time.",
            ],
        }
    }

    pub fn random_comment(&self) -> &str {
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.comments.len());
        self.comments[idx]
    }
}
