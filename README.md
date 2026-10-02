# FireProt

FireProt is an AMD GPU focused framework for protein structure inference. Its goal is to support models such as Protenix, AtlasFold, OpenFold, and ESMFold while preserving each model's scientific semantics and providing reproducible execution, measurement, and AMD operator optimization.

**Status:** the repository and a Protenix baseline are ready. The shared framework, additional model adapters, and optimized AMD operators have not been implemented. Existing smoke runs establish that three public Protenix models produce finite coordinates; they are not speed or accuracy benchmarks.

## Documentation

- [Architecture](docs/architecture.md): model adapters, execution, operator backends, validation, and implementation stages.
- [Protenix baseline notes](.agents/protenix_baseline.md): current cases, weights, remote environment, commands, and smoke results. These historical notes remain in Chinese.
- [Project context](.agents/project_context.md) and [agent instructions](AGENTS.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `Protenix/` | Independent upstream Git checkout; currently the only runnable model implementation |
| `cases/` | Eight single-chain development cases and experimental-coordinate mappings |
| `scripts/` | Protenix baseline inference, synchronization, and smoke checks |
| `provenance/` | Source, asset, case-hash, and smoke-test records |
| `.agents/` | Baseline history and project context |
| `docs/` | Framework architecture |

The top-level Git remote is `https://github.com/Media-Foundry/FireProt.git`. `Protenix/` retains its own Git history. The top-level repository excludes it, runtime assets in `runtime/`, and generated predictions in `outputs/`.
