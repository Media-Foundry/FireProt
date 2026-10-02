# Decision log

## 2026-10-02: Initialize the top-level repository

- The top-level repository manages project documentation, cases, scripts, and provenance. Its `origin` is `https://github.com/Media-Foundry/FireProt.git`.
- `Protenix/` keeps its independent upstream Git history. The top-level `.gitignore` excludes it to avoid an accidental gitlink.
- The top-level repository excludes `runtime/` and `outputs/`. Weights remain on DiamondHill; provenance records describe the existing smoke runs.
- `AGENTS.md` and `.agents/project_context.md` record collaboration and experimental conventions.

## 2026-10-02: Define the framework scope

- FireProt is an AMD GPU protein structure inference framework. Protenix is its first runnable baseline.
- The original README moved to `.agents/protenix_baseline.md`. The root README is the project entry point, and `docs/architecture.md` holds the design proposal.
- Model adapters and operator backends are separate extension axes. Shared contracts cover tasks, results, resources, and observation without forcing a common model-internal feature representation.
- Project-facing declarations are in English. The historical Protenix baseline notes remain in Chinese.
