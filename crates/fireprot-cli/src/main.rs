mod app;
mod args;
mod collection;
mod comparison;
mod inference;
mod support;

fn main() {
    if let Err(error) = app::run(std::env::args_os().skip(1)) {
        eprintln!("fireprot: {error}");
        std::process::exit(1);
    }
}
