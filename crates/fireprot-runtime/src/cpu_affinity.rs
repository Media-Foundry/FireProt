//! Explicit worker CPU placement without unsafe post-fork callbacks.
use nix::{
    sched::{CpuSet, sched_getaffinity, sched_setaffinity},
    unistd::Pid,
};
use std::{io, marker::PhantomData, rc::Rc};

pub(crate) const ENV: &str = "FIREPROT_WORKER_CPU_AFFINITY";

fn parse(setting: Option<&str>) -> io::Result<Option<CpuSet>> {
    let Some(setting) = setting.filter(|value| *value != "inherit") else {
        return Ok(None);
    };
    let invalid = || io::Error::new(io::ErrorKind::InvalidInput, "invalid worker CPU affinity");
    let number = |value: &str| -> io::Result<usize> {
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(invalid());
        }
        value.parse().map_err(|_| invalid())
    };
    let mut mask = CpuSet::new();
    for part in setting.split(',') {
        let (low, high) = part.split_once('-').unwrap_or((part, part));
        let (low, high) = (number(low)?, number(high)?);
        if low > high || high >= CpuSet::count() {
            return Err(invalid());
        }
        for cpu in low..=high {
            mask.set(cpu)?;
        }
    }
    Ok(Some(mask))
}

/// Only the calling Rust thread changes briefly. The child inherits its mask;
/// restore it immediately after spawn, before supervision or further work.
pub(crate) struct ScopedAffinity {
    original: Option<CpuSet>,
    calling_thread: PhantomData<Rc<()>>,
}

impl ScopedAffinity {
    pub(crate) fn enter(setting: Option<&str>) -> io::Result<Self> {
        let Some(requested) = parse(setting)? else {
            return Ok(Self {
                original: None,
                calling_thread: PhantomData,
            });
        };
        let original = sched_getaffinity(Pid::from_raw(0))?;
        for cpu in 0..CpuSet::count() {
            if requested.is_set(cpu)? && !original.is_set(cpu)? {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "worker CPU affinity must be a subset of the launch thread's allowed CPUs",
                ));
            }
        }
        sched_setaffinity(Pid::from_raw(0), &requested)?;
        let guard = Self {
            original: Some(original),
            calling_thread: PhantomData,
        };
        if sched_getaffinity(Pid::from_raw(0))? != requested {
            return Err(io::Error::other(
                "worker CPU affinity was not applied exactly",
            ));
        }
        Ok(guard)
    }

    pub(crate) fn restore(&mut self) -> io::Result<()> {
        if let Some(original) = self.original {
            sched_setaffinity(Pid::from_raw(0), &original)?;
            self.original = None;
        }
        Ok(())
    }
}

impl Drop for ScopedAffinity {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

pub(crate) fn verify(setting: Option<&str>, pid: u32) -> io::Result<()> {
    if let Some(expected) = parse(setting)? {
        let pid = i32::try_from(pid).map_err(|_| io::Error::other("invalid worker PID"))?;
        if sched_getaffinity(Pid::from_raw(pid))? != expected {
            return Err(io::Error::other(
                "worker CPU affinity disagrees with launch identity",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        os::unix::process::CommandExt,
        process::{Command, Stdio},
    };

    #[test]
    fn invalid_masks_and_bounds_are_rejected() {
        assert!(parse(None).unwrap().is_none());
        assert!(parse(Some("inherit")).unwrap().is_none());
        for value in [
            "",
            "-1",
            "1-0",
            "1-2-3",
            "0,",
            " 0",
            "+1",
            "x",
            "999999999999999999999",
        ] {
            assert!(parse(Some(value)).is_err(), "{value}");
        }
        assert!(parse(Some(&CpuSet::count().to_string())).is_err());
        let mask = parse(Some("0-2,2,4")).unwrap().unwrap();
        assert!(mask.is_set(0).unwrap() && mask.is_set(2).unwrap() && mask.is_set(4).unwrap());
        assert!(!mask.is_set(3).unwrap());
    }

    #[test]
    fn child_inherits_mask_and_parent_is_restored_on_success_and_failure() {
        let original = sched_getaffinity(Pid::from_raw(0)).unwrap();
        let cpu = (0..CpuSet::count())
            .find(|&cpu| original.is_set(cpu).unwrap())
            .unwrap();
        let setting = cpu.to_string();
        let mut guard = ScopedAffinity::enter(Some(&setting)).unwrap();
        let child = Command::new("/bin/sleep")
            .arg("10")
            .stdout(Stdio::null())
            .process_group(0)
            .spawn()
            .unwrap();
        let mut child = crate::command::ProcessGroup(child, false);
        guard.restore().unwrap();
        assert_eq!(sched_getaffinity(Pid::from_raw(0)).unwrap(), original);
        verify(Some(&setting), child.0.id()).unwrap();
        let other = ((cpu + 1) % CpuSet::count()).to_string();
        assert!(verify(Some(&other), child.0.id()).is_err());
        child.stop();
        {
            let _guard = ScopedAffinity::enter(Some(&setting)).unwrap();
            assert!(Command::new("/no-such-fireprot-worker").spawn().is_err());
        }
        assert_eq!(sched_getaffinity(Pid::from_raw(0)).unwrap(), original);
        if let Some(disallowed) = (0..CpuSet::count()).find(|&cpu| !original.is_set(cpu).unwrap()) {
            assert!(ScopedAffinity::enter(Some(&disallowed.to_string())).is_err());
            assert_eq!(sched_getaffinity(Pid::from_raw(0)).unwrap(), original);
        }
    }
}
