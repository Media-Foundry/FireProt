# Pending head-dimension 16 softmax coverage

Status: N1000/N1536 complete-model bitwise gates pass; no controlled speedup claim.
Do not deploy or run this trial during the active combined N1536 E2E sequence.

The accepted N1536 runtime records native attention selections for FP32
`[B,4,1536,16]` queries as well as HIP selections for dimension 32. Selection
records alone do not provide steady-state frequencies or latency. The maintained
adapter explicitly rejects head dimensions other than 32. The published softmax
C ABI consumes `[B,4,N,N]` scores and does not consume the QK reduction dimension.
This makes dimension 16 a candidate for additional coverage, subject to the
remaining mask/layout guards and whole-model validation.

`head16-adapter.patch` changes only an isolated copy of the adapter. Its base is
`python/fireprot/adapters/openfold3/attention.py`, SHA-256
`d8a884b9c018c436904a1e6774139e75915fae8c11189190425865a5780b9bd3`.
The patch accepts dimensions 16 and 32 and counts actual submitted softmax
launches separately for each dimension. It keeps native QK/PV einsums, precision,
mask/bias checks, score-layout fallback and source-pinned attention scope.
It does not change the maintained adapter or the published HIP library.

## Execution and acceptance

1. Wait for the active three-pair benchmark to terminate and finish its result
   checks/archive. Keep all active sources and binaries immutable until then.
2. Create a new frozen copy of the accepted all-in-place, pair/atom-cache worker
   with the measured N1536 confidence ceiling. Verify the adapter base hash,
   apply this patch in the copy, and record all source/request/profile pins.
3. Run fresh N1000 and N1536 numerical trials under the existing strict FP32,
   R3/four trunk passes, D200, S1, seed101 contract. Keep the original native
   references and require all 11 arrays bitwise. Concurrent trials are numerical
   evidence only; do not treat their durations as performance measurements.
4. Inspect `experimental_head_dimensions.submitted_launches` in the operator
   runtime record. Require actual dimension-16 launches, zero unexpected storage
   fallback, the pair/atom-cache counters and all final chunk-tuner selections.
   A changed memory footprint can change tuning and therefore floating-point
   results; preserve failed outputs and investigate rather than relax the gate.
5. Only after both model gates pass, measure the incremental improvement in an
   isolated paired comparison against the accepted dimension-32 combination.
   Do not multiply that incremental ratio into the original-native result.

Local preparation checks: the patch applies to the pinned base and the patched
Python source compiles. These checks do not establish numerical equivalence.
The patch adapts an Apache-2.0 file and remains Apache-2.0; retain the target's
existing copyright and license notices.

## First complete-model observation

N1000 completes and matches all11 original-native arrays bitwise. Its runtime
records264 actual D16 softmax submissions plus1702 D32 submissions, zero storage
fallbacks, and all module chunk tuners remain1024. Both conditioning caches
retain one miss and199 hits.

N1536 also completes and matches all11 original-native arrays bitwise. It records
400 D16 and2708 D32 submitted launches, zero storage fallback, and the same
conditioning-cache counts. Template/MSA/pairformer chunk tuners now select1024
(the dimension-32 combination selected512), while confidence remains256 and
diffusion conditioning1024. This specific change preserves exact outputs; it
does not establish arbitrary chunk sizes as interchangeable. Both completed
runs are archived in `provenance/openfold3_head16_models.json`. No paired
performance result or promotion is established yet.

For a same-source incremental comparison, `head16-control.patch` applies after
the adapter patch. It adds `FIREPROT_SOFTMAX_HEAD16=0/1`, rejecting other values;
the default1 preserves the tested D16 behavior. Both arms use the same adapter
bytes, kernel, caches and confidence ceiling. The comparison must declare the
flag and profile digest changes and inspect actual launch counts/tuner choices.
The paired control is prepared separately; this patch is not a maintained
production setting or a measured improvement by itself.
