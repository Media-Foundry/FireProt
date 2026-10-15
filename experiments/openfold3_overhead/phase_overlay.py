"""Create an isolated Rust source overlay for non-forward time attribution.

No validation is disabled. Events are wall-clock boundary timestamps, not summed
timers: worker and supervisor overlap. This is diagnostic instrumentation only.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(exist_ok=False)
    for name in ('Cargo.toml', 'Cargo.lock', 'LICENSE', 'crates'):
        path = source / name
        if path.is_dir():
            shutil.copytree(path, destination / name)
        else:
            shutil.copy2(path, destination / name)
    changes = []

    def edit(relative, replacements):
        path = destination / relative
        original = path.read_bytes()
        code = original.decode()
        for before, after in replacements:
            assert code.count(before) == 1, (relative, before, code.count(before))
            code = code.replace(before, after)
        code += '''
fn phase_boundary(label: &str) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).expect("clock before epoch");
    eprintln!("FIREPROT_PHASE {} {} {}", std::process::id(), now.as_nanos(), label);
}
'''
        path.write_text(code)
        changes.append(dict(path=relative, before=hashlib.sha256(original).hexdigest(),
                            after=hashlib.sha256(path.read_bytes()).hexdigest()))

    edit('crates/fireprot-cli/src/inference.rs', [
        ('    let prepared = prepare(request, origin)?;',
         '    phase_boundary("preflight_begin");\n    let prepared = prepare(request, origin)?;\n    phase_boundary("preflight_end");'),
        ('    let report = supervisor::run(', '    phase_boundary("supervisor_begin");\n    let report = supervisor::run('),
        ('    Ok(report)', '    phase_boundary("supervisor_end");\n    Ok(report)'),
    ])
    edit('crates/fireprot-runtime/src/supervisor.rs', [
        ('    let execution = ExecutionRecord::capture(', '    phase_boundary("capture_begin");\n    let execution = ExecutionRecord::capture('),
        ('    fs::write(\n        output.join("execution.json"),', '    phase_boundary("capture_end");\n    fs::write(\n        output.join("execution.json"),'),
        ('    let started = Instant::now();', '    phase_boundary("spawn_begin");\n    let started = Instant::now();'),
        ('    child.stop();', '    phase_boundary("drive_end");\n    child.stop();'),
        ('        execution\n            .verify(&worker.identity_files, &worker.identity_trees)', '        phase_boundary("post_verify_begin");\n        execution\n            .verify(&worker.identity_files, &worker.identity_trees)'),
        ('        if exit_code != Some(0)', '        phase_boundary("post_verify_end");\n        if exit_code != Some(0)'),
        ('        validate_content(&paths)?;', '        phase_boundary("content_begin");\n        validate_content(&paths)?;\n        phase_boundary("content_end");'),
        ('    if report.phase == Phase::Succeeded {', '    phase_boundary("seal_begin");\n    if report.phase == Phase::Succeeded {'),
        ('    write_report(&output, &report)?;', '    write_report(&output, &report)?;\n    phase_boundary("seal_end");'),
        ('                if session.phase() == Phase::AwaitingStart {', '                if session.phase() == Phase::AwaitingStart {\n                    phase_boundary("ready_received");'),
        ('                    monitor.verify_device(hip_device, session.device().unwrap())?;', '                    phase_boundary("prestart_verify_end");\n                    monitor.verify_device(hip_device, session.device().unwrap())?;'),
        ('                    session.authorize_start().map_err(|e| e.to_string())?;', '                    phase_boundary("start_authorized");\n                    session.authorize_start().map_err(|e| e.to_string())?;'),
    ])
    (destination / 'phase-overlay.json').write_text(json.dumps(dict(
        schema_version=1, source=str(source), changes=changes,
        semantics='Only stderr boundary events added; validation and arithmetic unchanged.',
        timing='Wall-clock boundaries; do not sum overlapping worker/supervisor intervals.'), indent=2) + '\n')


if __name__ == '__main__':
    prepare(*sys.argv[1:])
