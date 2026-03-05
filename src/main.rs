mod app;
mod brain;
mod bubble;
mod chat_input;
mod personality;
mod renderer;
mod state;
mod window;

fn main() {
    if let Err(e) = app::run() {
        eprintln!("Lanty crashed: {e}");
        std::process::exit(1);
    }
}
