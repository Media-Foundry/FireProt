//! Private build supervisor. Its stdin is an exclusive parent-lifetime socket.
//! Remain a process-group leader until the parent closes that socket or kills us.
use nix::{
    sys::signal::{Signal, killpg},
    unistd::{getpgrp, getpid},
};
use std::{
    fs,
    io::{self, Read},
    path::PathBuf,
    process::{Command, Stdio},
};

fn run() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 || getpid() != getpgrp() {
        return Err(io::Error::other(
            "compiler guard requires a private process group and four arguments",
        ));
    }
    let watcher = std::thread::spawn(|| {
        let mut byte = [0u8; 1];
        loop {
            match io::stdin().read(&mut byte) {
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                _ => {
                    let _ = killpg(getpgrp(), Signal::SIGKILL);
                    std::process::exit(1);
                }
            }
        }
    });
    let status = Command::new(&args[0])
        .arg(&args[1])
        .arg(&args[2])
        .stdin(Stdio::null())
        .status()?;
    let destination = PathBuf::from(&args[3]);
    let pending = destination.with_extension("pending");
    fs::write(
        &pending,
        serde_json::to_vec(&serde_json::json!({"success":status.success(),"code":status.code()}))?,
    )?;
    fs::rename(pending, destination)?;
    // Keep the PGID alive, including after compiler exit, until its owner reaps it.
    let _ = watcher.join();
    Err(io::Error::other("compiler lifetime watcher exited"))
}
fn main() {
    if let Err(error) = run() {
        eprintln!("fireprot compiler guard: {error}");
        std::process::exit(1);
    }
}
