# Accepted run archives and timing collection

Successful supervised runs now contain `accepted.json`. This schema-1 receipt
binds the exact bytes of `request.json`, `provenance.json`, `execution.json`,
`supervisor.json` and every worker output accepted by the supervisor. Metadata
is checked against the supervisor's original in-memory values. Outputs are
rehashed after content validation. Duplicate and reserved output paths fail.
The receipt is created exclusively; an existing file cannot be overwritten.

The receipt is written before the final supervisor report. Readers require both
files and matching hashes. Failure to create the receipt produces a failed
supervisor report. Interrupted writes or missing files cannot be collected.
The persisted supervisor interval is captured before archive sealing and final
report writing; it does not include those operations or initial preflight.

## Collect measurements

Save the SHA-256 of each completed `accepted.json` separately. A collection file
names those receipts, an explicit timing scope and a cache preparation statement:

```json
{
  "schema_version": 1,
  "scope": "model",
  "cache_state": "published kernels; fresh model process; filesystem cache not flushed",
  "runs": [
    {
      "path": "/path/to/completed-run/accepted.json",
      "sha256": "REPLACE_WITH_THE_RECORDED_64_CHARACTER_SHA256"
    }
  ]
}
```

```bash
target/release/fireprot collect-timings collection.json > timing-report.json
```

The command verifies the pinned receipt and every bound file. It requires a
successful supervisor result, positive isolation sample count, no recorded
failure, and consistent announced/physical device, PID and visibility records.
It then reruns model output-content validation, including archive arrays,
sample count, workload, seed, shapes, dtypes, finiteness and model confidence
requirements. This is content acceptance; numerical agreement with a reference
remains a separate correctness gate.

The supervisor archives support `model`, `worker_request` and `supervisor`.
`end_to_end` requires additional pinned records produced by
[`measure-inference`](invocation-timing.md), supplied as `invocations` in the
same order as `runs`. The collector rejects missing or mismatched records;
no supervisor interval is relabeled as E2E.

The collector builds a [timing series](timing-statistics.md) and its statistics.
It retains original requests and receipt pins, and rejects mixed conditions.
The comparison environment includes the full launch record, worker environment,
physical GPU mapping and source/resource provenance-file hash. Only the launch
digest and three invocation-specific environment values (`FIREPROT_REQUEST`,
`FIREPROT_OUTPUT_DIR`, `FIREPROT_WORKER_SOCKET`) are removed from the condition
comparison. A working directory exactly equal to the archived request's output
directory is represented as that relationship for comparison. Other fixed working
directories remain unchanged; no live-path resolution is needed for copied archives.
Different binaries, source inventories, settings, seeds, devices or
backends are not silently pooled. Every submitted run is included; duplicate
directories and run IDs fail.

## Portability and trust

An intact run directory can be copied. Content validation uses its new directory
while the result preserves the original request and recorded environment.
Collecting an archive does not need the original model checkout, weights, ROCm
or Python environment. It reads recorded identities and output arrays, not live
hardware. Historical runs without receipts remain historical evidence; no
automatic backfill can establish what their bytes were at acceptance time.

Receipts detect changes relative to a separately retained hash. They are not
signatures or proof against a writer who can replace both the archive and the
trusted receipt pin. Cache preparation is explicitly supplied by the caller and
cannot be reconstructed from a successful run. Continuous isolation remains an
observation made during inference, not something the collector can repeat later.
Full environment locking and paired performance verdicts remain unfinished.

## Verified integration (2026-10-04)

[The integration record](../provenance/rust_run_archive_integration.json) retains
receipt pins, accepted inventories, requests, independent device mappings,
collection outputs and numerical comparisons for four N1000 requests:

| Model / route | Requests | Archive and all three timing scopes | Numerical comparison |
| --- | ---: | --- | --- |
| AtlasFold + AtlasLM-3B / fused HIP | 2 | Passed individually and as a two-observation series on HIP 0 / PCI `0000:32:00.0` | All four public arrays exactly equal to the original native reference |
| Protenix Mini-default FP32 / FireProt | 1 | Passed on HIP 1 / PCI `0000:35:00.0` | Aligned RMSD 8.87182e-5 Angstrom and confidence max delta 3.81470e-6; both pass existing 1e-3 gates |
| OpenFold3 / native OpenBind-0 | 1 | Passed on HIP 2 / PCI `0000:11:00.0` | Exact comparison remains unpassed |

AtlasFold retains seed 102, four additional recycles, 30 diffusion steps and one
sample. Protenix retains seed 101, four total cycles, one step and one sample.
OpenFold3 retains seed 101, three additional recycles, 200 steps and one sample.
These requests test archive/collection integration; concurrent activity makes
their timings unsuitable for a new speedup claim. The kernels and model work
were unchanged.

The complete Rust suite passes 102 tests. Python passes 247 tests with 110 skips;
formatting and Clippy pass. Rust 1.85 passes the two collection and nine
supervisor tests. Failure coverage includes mutation of each protected metadata
file or output, receipt collision, failed/inconsistent supervision, corrupted
arrays, duplicate runs, invalid timing scope and relocated archives. Historical
receipts are not fabricated for earlier model runs.
