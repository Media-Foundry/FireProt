//! Parse the public positional command interface without running host services.
use std::path::Path;

pub enum Command<'a> {
    Help(Option<&'a str>),
    Version,
    BenchmarkPairs {
        specification: &'a Path,
        destination: &'a Path,
    },
    CompareRuns(&'a Path),
    MeasureInference {
        specification: &'a Path,
        destination: &'a Path,
    },
    Plm {
        family: &'a str,
        request: &'a Path,
        profile: &'a Path,
    },
    InferBatch(&'a Path),
    InspectCif(&'a Path),
    CompareProtenixCif {
        reference: &'a Path,
        reference_confidence: &'a Path,
        candidate: &'a Path,
        candidate_confidence: &'a Path,
    },
    CaptureEnvironment(&'a Path),
    Install {
        specification: &'a Path,
        destination: &'a Path,
    },
    VerifyInstallation {
        directory: &'a Path,
        digest: &'a str,
    },
    Inspect(&'a Path),
    Infer {
        request: &'a Path,
        profile: &'a Path,
    },
    SummarizeTimings(&'a Path),
    CollectTimings(&'a Path),
    ExportSource {
        root: &'a Path,
        commit: &'a str,
    },
    CompareNpz {
        reference: &'a Path,
        candidate: &'a Path,
    },
    BuildTorch {
        specification: &'a Path,
        cache: &'a Path,
    },
    BuildHip {
        specification: &'a Path,
        cache: &'a Path,
    },
    PublishKernel {
        cache: &'a Path,
        destination: &'a Path,
    },
    VerifyKernel {
        directory: &'a Path,
        digest: &'a str,
        compatibility: &'a Path,
    },
}

const COMMANDS: &[(&str, &str, &str)] = &[
    (
        "esm",
        "REQUEST EXECUTION",
        "Run standalone ESMC, ESM2 or ESM3 sequence inference.",
    ),
    (
        "atlaslm",
        "REQUEST EXECUTION",
        "Run standalone AtlasLM sequence inference.",
    ),
    (
        "benchmark-pairs",
        "SPECIFICATION DESTINATION",
        "Execute pinned pairs with per-run correctness gates and recorded ordering.",
    ),
    (
        "compare-runs",
        "SPECIFICATION",
        "Verify paired archives, declared treatment changes, numerical gates and timing ratios.",
    ),
    (
        "measure-inference",
        "SPECIFICATION DESTINATION",
        "Measure a complete pinned inference invocation and bind its accepted archive.",
    ),
    (
        "infer-batch",
        "MANIFEST",
        "Run pinned requests sequentially, stopping on failure or cancellation.",
    ),
    (
        "inspect-cif",
        "STRUCTURE",
        "Validate a single-model prediction CIF and report its atom count.",
    ),
    (
        "compare-protenix-cif",
        "REFERENCE_CIF REFERENCE_CONFIDENCE CANDIDATE_CIF CANDIDATE_CONFIDENCE",
        "Apply established Protenix coordinate and confidence gates to CIF predictions.",
    ),
    (
        "capture-environment",
        "SPECIFICATION",
        "Hash declared model environment files and trees into a reusable inventory.",
    ),
    (
        "install",
        "SPECIFICATION DESTINATION",
        "Install framework files and generate execution profiles in a new directory.",
    ),
    (
        "verify-installation",
        "DIRECTORY MANIFEST_SHA256",
        "Verify all installed files against a pinned manifest.",
    ),
    (
        "inspect",
        "REQUEST",
        "Validate a request without opening weights or using a GPU.",
    ),
    (
        "infer",
        "REQUEST EXECUTION",
        "Run a pinned model request with process and GPU supervision.",
    ),
    (
        "summarize-timings",
        "SERIES_JSON",
        "Summarize observations with matching measurement conditions.",
    ),
    (
        "collect-timings",
        "COLLECTION_JSON",
        "Reverify accepted run archives and collect recorded timings.",
    ),
    (
        "export-source",
        "ROOT COMMIT",
        "Verify and export a source manifest for the pinned Git commit.",
    ),
    (
        "compare-npz",
        "REFERENCE CANDIDATE",
        "Compare all arrays exactly, including shape and storage dtype.",
    ),
    (
        "build-torch",
        "SPECIFICATION CACHE",
        "Build or reuse a verified Torch/HIP extension.",
    ),
    (
        "build-hip",
        "SPECIFICATION CACHE",
        "Build or reuse a verified standalone HIP library.",
    ),
    (
        "publish-kernel",
        "CACHE DESTINATION",
        "Publish a verified kernel artifact to a new directory.",
    ),
    (
        "verify-kernel",
        "DIRECTORY MANIFEST_SHA256 COMPATIBILITY_JSON",
        "Verify a pinned kernel artifact and runtime compatibility.",
    ),
];

pub fn help(command: Option<&str>) -> String {
    match command {
        Some(name) => {
            let (_, operands, description) = COMMANDS
                .iter()
                .find(|(n, _, _)| *n == name)
                .expect("validated help command");
            format!("Usage: fireprot {name} {operands}\n\n{description}\n")
        }
        None => {
            let mut text = String::from(
                "FireProt — AMD protein structure inference\n\nUsage: fireprot COMMAND [ARGUMENTS]\n\nCommands:\n",
            );
            for (name, _, description) in COMMANDS {
                text.push_str(&format!("  {name:<19} {description}\n"));
            }
            text.push_str("\nOptions:\n  -h, --help       Show help.\n  -V, --version    Show the framework version.\n\nUse 'fireprot help COMMAND' or 'fireprot COMMAND --help' for arguments.\n");
            text
        }
    }
}

fn known(name: &str) -> bool {
    COMMANDS.iter().any(|(n, _, _)| *n == name)
}

pub fn parse(args: &[String]) -> Result<Command<'_>, String> {
    let words: Vec<_> = args.iter().map(String::as_str).collect();
    Ok(match words.as_slice() {
        ["--help" | "-h" | "help"] => Command::Help(None),
        ["--version" | "-V"] => Command::Version,
        ["help", name] | [name, "--help" | "-h"] if known(name) => Command::Help(Some(*name)),
        ["benchmark-pairs", specification, destination] => Command::BenchmarkPairs {
            specification: Path::new(*specification),
            destination: Path::new(*destination),
        },
        ["compare-runs", path] => Command::CompareRuns(Path::new(*path)),
        ["measure-inference", specification, destination] => Command::MeasureInference {
            specification: Path::new(*specification),
            destination: Path::new(*destination),
        },
        ["infer-batch", path] => Command::InferBatch(Path::new(*path)),
        ["inspect-cif", path] => Command::InspectCif(Path::new(*path)),
        ["compare-protenix-cif", r, rc, c, cc] => Command::CompareProtenixCif {
            reference: Path::new(*r),
            reference_confidence: Path::new(*rc),
            candidate: Path::new(*c),
            candidate_confidence: Path::new(*cc),
        },
        ["capture-environment", path] => Command::CaptureEnvironment(Path::new(*path)),
        ["install", specification, destination] => Command::Install {
            specification: Path::new(*specification),
            destination: Path::new(*destination),
        },
        ["verify-installation", directory, digest] => Command::VerifyInstallation {
            directory: Path::new(*directory),
            digest,
        },
        ["inspect", request] => Command::Inspect(Path::new(*request)),
        [family @ ("esm" | "atlaslm"), request, profile] => Command::Plm {
            family,
            request: Path::new(*request),
            profile: Path::new(*profile),
        },
        ["infer", request, profile] => Command::Infer {
            request: Path::new(*request),
            profile: Path::new(*profile),
        },
        ["summarize-timings", path] => Command::SummarizeTimings(Path::new(*path)),
        ["collect-timings", path] => Command::CollectTimings(Path::new(*path)),
        ["export-source", root, commit] => Command::ExportSource {
            root: Path::new(*root),
            commit,
        },
        ["compare-npz", reference, candidate] => Command::CompareNpz {
            reference: Path::new(*reference),
            candidate: Path::new(*candidate),
        },
        ["build-torch", specification, cache] => Command::BuildTorch {
            specification: Path::new(*specification),
            cache: Path::new(*cache),
        },
        ["build-hip", specification, cache] => Command::BuildHip {
            specification: Path::new(*specification),
            cache: Path::new(*cache),
        },
        ["publish-kernel", cache, destination] => Command::PublishKernel {
            cache: Path::new(*cache),
            destination: Path::new(*destination),
        },
        ["verify-kernel", directory, digest, compatibility] => Command::VerifyKernel {
            directory: Path::new(*directory),
            digest,
            compatibility: Path::new(*compatibility),
        },
        [] => return Err("missing command; run 'fireprot --help'".into()),
        ["help", name] => return Err(format!("unknown command '{name}'; run 'fireprot --help'")),
        [name, ..] if known(name) => {
            return Err(format!(
                "invalid arguments\n{}",
                help(Some(name)).trim_end()
            ));
        }
        [name, ..] => return Err(format!("unknown command '{name}'; run 'fireprot --help'")),
    })
}
