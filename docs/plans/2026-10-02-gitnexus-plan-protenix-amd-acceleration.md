# GitNexus Engineering Plan

> Task: Analyze Protenix v1.1.0, locate AMD GPU inference hotspots, and define an implementation-ready operator acceleration plan that can extend to future model adapters.
> Evidence verified at top-level commit ac319e9234eb91dbac390ca91a2dccbfac8b88a2 and independent Protenix commit d3b4db6a121dd4584edd85e93744239325a2b72e; GitNexus index refreshed this session with --index-only --pdg using GitNexus 1.6.12.
> Evidence provenance schema 2; global dirty digest 0a9c85780067d9afcd0764f307b60891e3cee927ee11eaeb5ec7826d10fd82cd; cited-path manifest 34 sorted entries; exact generated plan path excluded.

## 1. Objective

Create a measurement-gated AMD acceleration path for the current Protenix inference baseline. The implementation should improve warm inference latency on MI250-class gfx90a GPUs, preserve Protenix FP32 behavior, and establish operator contracts that AtlasFold, OpenFold, ESMFold, and later adapters can reuse.

This plan separates three decisions: what Protenix computes, which semantic operations FireProt exposes, and which AMD implementation runs for a supported tensor signature. No optimization is promoted from the exploratory measurements alone.

## 2. Current Behaviour

[verified] The native entry path is runner.batch_inference.predict → inference_jsons → InferenceRunner.predict → Protenix.forward. The runner constructs and moves the model to the selected device, applies the requested autocast policy, preprocesses each input, runs inference, and serializes results (Protenix/runner/batch_inference.py:421-535, 744-930; Protenix/runner/inference.py:64-236, 410-535).

[verified] Protenix.forward dispatches inference to main_inference_loop and _main_inference_loop. The latter runs get_pairformer_output, diffusion sampling, distogram work, and confidence prediction in that order (Protenix/protenix/model/protenix.py:349-630, 843-930).

[verified] The Mini configuration uses four recycling cycles, sixteen Pairformer blocks, one MSA block, and an eight-block diffusion transformer; the configured inference diffusion-step default is five (Protenix/configs/configs_model_type.py:153-220).

[verified] Each PairformerBlock applies outgoing triangle multiplication, incoming triangle multiplication, starting-node triangle attention, a transpose/contiguous conversion, ending-node triangle attention, a second transpose/contiguous conversion, and the pair transition. When a single representation exists, attention-with-pair-bias and a single transition follow (Protenix/protenix/model/modules/pairformer.py:103-224).

[verified] The native triangle multiplication path performs LayerNorm, four linear projection/gating operations, masked batched matrix multiplication, output normalization, output projection, output gating, and residual addition. Its inference path also chunks work for memory control (Protenix/protenix/model/triangular/triangular.py:199-570).

[verified] Triangle attention normalizes pair features, constructs mask and triangle biases, projects Q/K/V, dispatches an attention backend, gates and projects the result, and returns to Pairformer. The torch backend materializes attention logits and softmax; the existing triattention option calls a Triton online-softmax implementation (Protenix/protenix/model/triangular/triangular.py:589-715; Protenix/protenix/model/triangular/layers.py:299-473; Protenix/protenix/model/tri_attention/op.py:36-190).

[verified] Diffusion sampling iterates the noise schedule and calls the denoiser for every step. DiffusionModule encodes atoms, explicitly upcasts the token state and conditioning to FP32, applies the diffusion transformer, and decodes atom-coordinate updates (Protenix/protenix/model/generator.py:119-255; Protenix/protenix/model/modules/diffusion.py:325-600).

[verified] The runner's existing “Model forward time” boundary includes output dumping, and internal model timers use host wall time without an explicit device synchronization. These values cannot be used as GPU stage latency (Protenix/runner/inference.py:410-535; Protenix/protenix/model/protenix.py:463-630).

## 3. Relevant Architecture

[verified] FireProt's proposed architecture already separates model adapters, model-neutral operator contracts, AMD backends, runtime selection, validation, tests, and benchmarks (docs/architecture.md:1-87). The implementation should preserve this ownership:

1. The Protenix adapter maps Protenix tensors, flags, and output semantics to FireProt contracts.
2. The operation registry describes semantic operations and selects only implementations whose capability predicates match.
3. AMD backends implement kernels using PyTorch/hipBLAS, hipBLASLt, CK, Triton, or HIP without exposing those libraries to model adapters.
4. Native Protenix remains the reference implementation and automatic fallback.
5. Benchmark and validation code records the implementation selected for every operation signature.

[inferred] Pair representations have shape proportional to N², and the triangle contractions and attention grow much faster with sequence length than token-linear work. The measured 124-token mix therefore ranks the current short-case hotspots but does not predict Base-model or long-sequence shares.

[verified] Protenix is an independent checkout. Changes under Protenix/ must be reviewed and committed in that repository; the top-level repository must not record it as a gitlink. Runtime weights, CCD resources, output predictions, and hardware-local tuning caches remain untracked (.agents/project_context.md:1-25; README.md:1-24).

## 4. GitNexus Findings

- [graph] mcp__gitnexus__query(search_query="batch inference prediction model forward Pairformer diffusion sampling confidence output") located runner.batch_inference.predict at line 744, Protenix at protenix/model/protenix.py:91, and sample_diffusion at generator.py:119. Key result: “predict …:744; Protenix …:91; sample_diffusion …:119.”
- [graph] mcp__gitnexus__query(search_query="PairformerStack triangle multiplication triangle attention diffusion transformer AtomAttention forward") located PairformerBlock.forward at line 103, DiffusionModule.forward at line 508, the shared Attention implementation, AtomTransformer, and related tests. Key result: “PairformerBlock.forward:103; DiffusionModule.forward:508; Attention:299; AtomTransformer:471.”
- [graph] mcp__gitnexus__context on predict, Protenix.forward, sample_diffusion, PairformerBlock.forward, and DiffusionModule.forward confirmed the high-level CLI and model calls but missed dynamic nn.Module invocation edges. Key result: “Protenix.forward calls main_inference_loop; DiffusionModule.forward delegates to f_forward.”
- [graph] mcp__gitnexus__impact(direction="upstream", maxDepth=3, summaryOnly=true) reported zero resolved callers and risk UNKNOWN for the four model-internal primary symbols. Key result: “impactedCount: 0; risk: UNKNOWN; confirm with text search.” This is a graph limitation, not evidence that the symbols have no callers.
- [verified] Source verification accounts for the dynamic call chain through PairformerStack blocks, triangle modules, the repeated diffusion denoiser, and confidence paths. Any executor must use source searches around these module calls when editing dispatch.
- [verified] Existing tests for AttentionPairBias, conditioned transitions, and the diffusion transformer mainly assert shapes. The fused dropout-add suite covers fallback, gradients, and Pairformer contiguity. No dedicated triangle multiplication or numerical triangle-attention equivalence test exists (Protenix/tests/test_attention_pair_bias.py:1-125; Protenix/tests/test_condition_transition.py:1-70; Protenix/tests/test_fused_dropout_add.py:1-260; Protenix/tests/test_diffusion_transformer.py:1-260; Protenix/tests/test_triton_compatibility.py:1-180).
- [verified] The focused existing-test command was collected on DiamondHill with the fold Python environment: 18 tests across the five named files.

## 5. Statement-Level PDG Findings

### PairformerBlock.forward

[graph] mcp__gitnexus__pdg_query(mode="flows", target="protenix/model/modules/pairformer.py", variable="z", limit=200) returned 38 file-level edges, with the relevant function anchored at line 103.

- z flows through two control-dependent implementations: the inference-oriented inplace-safe path at line 139 and the residual/dropout path at line 171.
- Both paths preserve the same operation order before z feeds attention_pair_bias and the block return at lines 218 and 224.
- The two explicit transpose(...).contiguous() operations are semantic layout transitions between starting-node and ending-node attention, so removing them requires an attention kernel that accepts both orientations or a proved equivalent view/stride contract.
- Planning consequence: the first custom kernels are inference-only, replace one semantic operation at a time, and fall back for training, autograd, unsupported strides, unsupported masks, or ambiguous aliasing.

### DiffusionModule.f_forward

[graph] mcp__gitnexus__pdg_query(mode="flows", target="protenix/model/modules/diffusion.py", variable="a_token", limit=200) returned 12 relevant edges in the function anchored at line 325.

- a_token is defined by the atom encoder branches, explicitly converted to FP32 at line 454, augmented with normalized single conditioning at lines 458/462, transformed at line 470, and consumed by the atom decoder at lines 483/496.
- Planning consequence: preserve the explicit FP32 transformer boundary. A later mixed-precision diffusion implementation is a separate accuracy project, not an incidental kernel optimization.

### Forward dispatch and control

[graph] The control slices for Protenix.forward and PairformerBlock.forward show mode and inplace_safe as load-bearing predicates. Inference dispatch reaches main_inference_loop, while training/evaluation and non-inplace Pairformer paths have different memory and autograd requirements.

- Planning consequence: every accelerated branch must state supported mode, dtype, device architecture, shape, layout, mask, and aliasing behavior. A false capability predicate must call the existing implementation with no output-schema change.

## 6. Proposed Changes

### A. Establish a reproducible performance contract

**Files:** benchmarks/protenix/profile_inference.py; benchmarks/protenix/configs/mi250.yaml
**Existing integration symbols:** InferenceRunner.predict, Protenix._main_inference_loop, PairformerBlock.forward, DiffusionModule.forward.

Add a harness that records model/checkpoint identity, input identity and token/atom counts, dtype, cycle count, diffusion steps, sample count, seed, cache state, device/ROCm/PyTorch versions, GPU clocks, selected backend, warmup count, measured repetitions, and exclusive-device status. Use device events plus explicit synchronization for stage timing; keep preprocessing and serialization outside GPU model latency. Emit raw per-run JSON and summaries, and import rocprof kernel statistics rather than parsing human log lines.

The initial matrix must include Tiny/Mini and later Base when assets allow; token-length buckets around 64, 124, 256, 512, and the longest feasible production case; S1, S5, and the production diffusion-step count; FP32 first; cold and warm cache states; at least five warmups and thirty measured repetitions per promoted bucket.

### B. Add model-neutral operation selection

**Files:** fireprot/ops/registry.py; fireprot/backends/amd/blas.py; fireprot/adapters/protenix/runtime.py; scripts/infer.sh.

Create an operation registry keyed by semantic operation, direction/layout, dtype, architecture, dimensions, mask form, contiguity, mode, and aliasing support. Each candidate declares a capability predicate, priority, version, and native fallback. Cache selection by the complete signature and emit the selected implementation in the run manifest.

Expose process-start BLAS policy values native, hipblaslt, hipblas, and ck for controlled experiments. Never switch PyTorch's process-global preferred BLAS library inside individual model calls. Per-shape routing belongs inside registered custom operations; unsupported CK problems fall back automatically.

The Protenix runtime adapter installs registrations before model construction, passes explicit backend choices to the native CLI, and leaves current defaults unchanged when acceleration is disabled.

### C. Implement triangle multiplication first

**Files:** fireprot/backends/amd/triangle_multiplication.py; Protenix/protenix/model/triangular/triangular.py; tests/operators/test_triangle_multiplication.py.

Add an FP32 inference implementation for outgoing and incoming TriangleMultiplicativeUpdate. Preserve mask application, projection/gate weights, accumulation precision, output LayerNorm/projection/gate, residual behavior, and inplace-safe semantics.

Implement in two measured increments:

1. Route the observed GEMM/BMM signatures to the fastest validated hipBLAS, hipBLASLt, or CK implementation while retaining native elementwise operations.
2. Fuse profitable adjacent work: input LayerNorm with projection/gating setup, mask application, and the output LayerNorm/projection/gate/residual epilogue. Keep contraction GEMMs library-backed unless a fused kernel wins after workspace and launch costs.

Promotion rule: register only signatures whose repeated operator benchmark has at least 1.10× median speedup and no p95 regression versus native. Other signatures stay native.

### D. Implement triangle attention second

**Files:** fireprot/backends/amd/triangle_attention.py; Protenix/protenix/model/triangular/triangular.py; Protenix/protenix/model/triangular/layers.py; tests/operators/test_triangle_attention.py.

Implement forward-only FP32 online softmax that consumes projected Q/K/V plus mask and triangle biases without materializing the full logits/softmax tensor. Support both starting-node and ending-node orientation so the adapter can later eliminate one or both contiguous copies.

Evaluate the existing Triton algorithm on gfx90a as a candidate, not as a presumed solution. Compare AMD-compatible Triton against a CK/HIP implementation for compilation time, workspace, kernel count, numerical agreement, and end-to-end latency. Keep the torch attention path for unsupported head dimensions, layouts, masks, architectures, dtypes, and all training/autograd calls.

Remove transpose/contiguous materialization only after a stride-aware kernel passes direct equivalence tests for both orientations.

### E. Reduce launch, allocation, and transfer overhead after triangle kernels

**Files:** fireprot/backends/amd/triangle_multiplication.py; fireprot/backends/amd/triangle_attention.py; fireprot/adapters/protenix/runtime.py.

Hoist invariant masks and scalar tensors, keep the diffusion schedule device-resident, reuse bounded temporary buffers where aliasing permits, and fuse LayerNorm/activation/gate/residual epilogues only where profiles still show material cost. Add allocation count, H2D copy count, and launch count to benchmark output.

Evaluate HIP graph capture only for stable, bucketed shapes after operator dispatch is fixed. Graph capture must include seed/state handling and fall back for dynamic shapes or unsupported allocations.

### F. Validate and document promotion

**Files:** tests/integration/test_protenix_numerics.py; README.md; docs/architecture.md.

Compare native and accelerated FP32 outputs with the same weights, input, seed, cycles, diffusion steps, samples, and cache state. Lock operator tolerances before broad benchmarking. The initial end-to-end gate is aligned coordinate RMSD ≤ 1e-3 Å and maximum absolute numeric confidence delta ≤ 1e-3 versus native for the profiled deterministic cases; any failure blocks promotion and triggers operator-level localization.

Run all eight development cases for regressions, clearly label them as development cases, and use only coordinates[mask] for experimental-coordinate metrics. Update documentation only with reproduced results and include run artifact paths and complete provenance.

## 7. Implementation Sequence

1. **Land the measurement harness in the top-level repository.** Capture native cold/warm runs, per-stage device time, operator shapes, kernel classes, launches, allocations, and transfers. Stop if repeated results are unstable or metadata is incomplete.
2. **Lock numerical baselines.** Produce same-seed native outputs for C4/S1 and C4/S5 across the selected development cases; define operator and end-to-end tolerances before adding custom kernels.
3. **Add the registry, process-start BLAS policy, and Protenix bootstrap.** Default to native. Verify backend selection manifests, unsupported-backend errors, and automatic fallback.
4. **Run the BLAS matrix.** Benchmark hipBLASLt, hipBLAS, and CK for every captured GEMM/BMM signature, including first-use and workspace cost. Generate a hardware/software-fingerprinted table in the untracked runtime cache.
5. **Implement and promote triangle multiplication signatures.** First library routing, then measured fusion. Commit nested Protenix dispatch changes separately from top-level backend code.
6. **Implement and promote triangle attention signatures.** Validate online softmax, both orientations, masks, and fallback before attempting layout-copy removal.
7. **Reprofile and address residual overhead.** Apply only fusions, invariant hoisting, buffer reuse, or graph capture justified by the new profile.
8. **Run end-to-end numerical and performance gates.** Use exclusive MI250 execution, fixed clocks where operationally available, five warmups, thirty repetitions, median/p95 reporting, and paired native/accelerated ordering.
9. **Regenerate benchmark reports once at the tip and update English documentation.** Do not encode exploratory numbers as results. Commit top-level changes in FireProt and model-source changes in the independent Protenix checkout.

## 8. Test Strategy

### Operator correctness

- Triangle multiplication: N={1,16,31,32,62,124,127,128,256}; incoming/outgoing; full, partial, and empty masks; contiguous and noncontiguous inputs; inplace-safe true/false; unsupported dtype/layout; CK unsupported problem; finite and close FP32 output.
- Triangle attention: starting/ending orientation; head/tile boundary lengths; both biases; absent/full/sparse masks; all-masked rows; extreme logits; noncontiguous layouts; unsupported architecture/dtype/head dimension; finite online-softmax output close to native.
- Registry: deterministic priority, exact cache key, implementation version invalidation, capability failure, native fallback, and manifest reporting.
- BLAS policy: selection occurs before model construction, invalid values fail clearly, and no per-call global mutation occurs.

### Integration and scientific regression

- Compare native and accelerated Mini-default for 1izr and 1izq at C4/S1 and C4/S5 with fixed seed 101, FP32, one sample, identical cache state.
- Run all eight development cases as regression coverage. Keep accuracy claims limited because these cases are not independent.
- Verify output keys, coordinate finiteness, confidence fields, and deterministic same-seed behavior.
- For any experimental-coordinate metric, slice coordinates[mask]; never score the input reference conformation as ground truth.
- Keep BF16 disabled until separate distribution-level structure and confidence validation passes.

### Performance gates

- Operator promotion: ≥1.10× median speedup over native for each registered signature, with no p95 regression, after warmup.
- End-to-end promotion: ≥1.05× median warm-latency speedup for every advertised bucket; report p95, peak memory, launches, allocations, and H2D copies. A regression in any advertised bucket removes that dispatch key or blocks release.
- Cold-start reporting is separate and includes MIOpen/library compilation/cache state.
- Do not enforce noisy hardware performance thresholds in ordinary CPU CI. Run them on the named DiamondHill MI250 profile job and retain raw JSON.

### Verification commands

~~~bash
cd Protenix
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_attention_pair_bias.py \
  tests/test_condition_transition.py \
  tests/test_diffusion_transformer.py \
  tests/test_fused_dropout_add.py \
  tests/test_triton_compatibility.py

cd ..
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/operators/test_triangle_multiplication.py \
  tests/operators/test_triangle_attention.py \
  tests/integration/test_protenix_numerics.py

python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250.yaml --backend native
python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250.yaml --backend fireprot
rocprofv3 --stats -- python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250.yaml \
  --backend fireprot --profile-one
~~~

The existing-test command was verified to collect 18 tests on DiamondHill. The new-test and benchmark commands become runnable when their planned files land.

## 9. Risk and Impact Analysis

### Measured hotspot evidence

[verified] In a controlled planning profile on one DiamondHill gfx90a GCD, Mini-default, 124 tokens, FP32, C4/S1, seed 101, one sample, no MSA/template, and torch triangle kernels produced 17,995 GPU kernel dispatches. rocprof grouped 412.753 ms of aggregate kernel duration into GEMM 40.76%, elementwise 19.57%, LayerNorm 18.72%, copy/transpose 18.01%, softmax 2.14%, and reductions 0.46%.

[verified] The paired torch profile recorded 1,851 H2D copies taking 53.602 ms, 13,994 aten::empty calls, 18.31 GB cumulative allocation, and 10,398 hipLaunchKernel calls consuming 1.838 s of host time. These counts justify treating launch, allocation, and small-transfer overhead as a second-order target.

[verified] For the second case in one C4/S5 FP32 process, device-event stage timing measured approximately 474.685 ms Pairformer, 100.407 ms diffusion, and 28.771 ms confidence with hipBLASLt. This makes Pairformer roughly 74% of the measured warm total for this short-case configuration.

[verified] A single exploratory global-hipBLAS run measured approximately 0.57 s warm total versus 0.64 s with hipBLASLt and reduced aggregate triangle-multiplication event time from 452.425 ms to 347.303 ms. This supports a controlled backend experiment; one warm observation does not establish a released speedup.

[verified] Shape microbenchmarks had mixed winners across hipBLASLt, hipBLAS, and CK, and CK rejected at least one observed BF16 BMM. A single global library policy cannot be the final selector.

[verified] FP32 hipBLAS versus hipBLASLt stayed close on the two profiled cases: aligned coordinate RMSD ≤0.000107 Å and maximum numeric confidence delta ≤0.000035. Current BF16 runs were slower and one case diverged by more than 8 Å after alignment, so BF16 is excluded from the first promotion.

These measurements are ephemeral planning evidence and must be reproduced by the planned harness before publication.

### Correctness and compatibility

- **High risk:** Triangle multiplication and attention sit inside every Pairformer block and recycling cycle; small reduction differences can amplify through diffusion.
- **High risk:** Layout removal changes strides and orientation. Test start/end semantics independently before deleting contiguous copies.
- **High risk:** Process-global BLAS preference affects every GEMM. Set it once during bootstrap and record it.
- **Medium risk:** Custom ops may compile or allocate on first use, worsening cold latency. Record compilation and warmup separately.
- **Medium risk:** CK capability gaps vary by shape/dtype. Dispatch must test support before invocation.
- **Medium risk:** The existing Triton kernel was not established as gfx90a-compatible or faster; its math is a reference candidate only.
- **Low risk when gated:** Native fallback retains existing output schema and behavior.

### Impact accounting

[graph] GitNexus reported no direct depth-1 dependents for the primary methods because dynamic nn.Module calls were unresolved. The plan therefore accounts for direct source consumers explicitly: PairformerBlock → TriangleMultiplicativeUpdate/TriangleAttention; TriangleAttention → Attention; PairformerStack and template/confidence stacks → PairformerBlock; _main_inference_loop → get_pairformer_output/sample_diffusion/confidence; generator.sample_diffusion → diffusion_module.

Future model adapters consume only FireProt operation contracts. They must provide model-specific semantic mapping and validation; they must not inherit Protenix masks, layouts, weights, or tolerances by assumption.

## 10. Files Expected to Change

| File | Symbols | Reason |
| ---- | ------- | ------ |
| benchmarks/protenix/profile_inference.py | new module | Reproducible stage/operator profiling and result manifests |
| benchmarks/protenix/configs/mi250.yaml | new configuration | Initial measurement matrix and promotion protocol |
| fireprot/ops/registry.py | new module | Model-neutral capability and fallback registry |
| fireprot/backends/amd/blas.py | new module | Explicit BLAS policy and signature benchmark records |
| fireprot/backends/amd/triangle_multiplication.py | new module | AMD triangle multiplication implementation |
| fireprot/backends/amd/triangle_attention.py | new module | AMD online-softmax triangle attention |
| fireprot/adapters/protenix/runtime.py | new module | Protenix bootstrap and semantic mapping |
| Protenix/runner/batch_inference.py | predict | Accept explicit FireProt backend selection |
| Protenix/protenix/model/triangular/triangular.py | TriangleMultiplicativeUpdate.forward; TriangleAttention.forward | Capability-gated custom-op dispatch |
| Protenix/protenix/model/triangular/layers.py | Attention.forward | Attention implementation dispatch |
| scripts/infer.sh | shell entry point | Route accelerated mode through the adapter |
| tests/operators/test_triangle_multiplication.py | new tests | Operator numerical and fallback coverage |
| tests/operators/test_triangle_attention.py | new tests | Attention numerical and fallback coverage |
| tests/integration/test_protenix_numerics.py | new tests | Same-seed end-to-end regression |
| README.md | usage documentation | Backend selection and measured-status language |
| docs/architecture.md | architecture documentation | Final registry, adapter, and fallback contracts |

## 11. Reusable Implementation Context

~~~json
{
  "implementation_context": {
    "task_summary": "Build a measurement-gated AMD acceleration path for Protenix v1.1.0 while defining reusable operator and backend boundaries for future protein-folding adapters.",
    "acceptance_criteria": [
      "Reproduce cold and warm stage, operator, launch, allocation, and kernel-shape measurements with complete run metadata.",
      "Preserve native Protenix FP32 semantics through capability-gated accelerated operators and an automatic native fallback.",
      "Select BLAS or custom implementations per operation signature rather than asserting one global backend is universally fastest.",
      "Promote an optimized bucket only after numerical and repeated MI250 performance gates pass."
    ],
    "evidence_provenance": {
      "schema_version": 2,
      "head_commit": "ac319e9234eb91dbac390ca91a2dccbfac8b88a2",
      "generated_plan_path": "docs/plans/2026-10-02-gitnexus-plan-protenix-amd-acceleration.md",
      "global_dirty_digest": {
        "algorithm": "sha256",
        "canonicalization": "gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records",
        "value": "0a9c85780067d9afcd0764f307b60891e3cee927ee11eaeb5ec7826d10fd82cd"
      },
      "cited_path_manifest": [
        {
          "path": ".agents/project_context.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:2d3fca5f3fd6721528ce55549c2b1dd8221f4068a1841012e52a0486c31ac8f4",
          "index_digest": "sha256:2d3fca5f3fd6721528ce55549c2b1dd8221f4068a1841012e52a0486c31ac8f4",
          "worktree_digest": "sha256:2d3fca5f3fd6721528ce55549c2b1dd8221f4068a1841012e52a0486c31ac8f4",
          "untracked_digest": "absent"
        },
        {
          "path": "Protenix/configs/configs_model_type.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:d7f59f1b2d83b3657555893be085502b04c38b21a069e8ebb4b05229a8f12552"
        },
        {
          "path": "Protenix/protenix/model/generator.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:84338db0cda852d859e498c9aee972747a319e4e701cfae271ac882215fe25ae"
        },
        {
          "path": "Protenix/protenix/model/modules/diffusion.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:c92001aeb046ea9a5dbbc8120a5487c76d0497c52ff7fe178143f443dd4cc988"
        },
        {
          "path": "Protenix/protenix/model/modules/fused_ops.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:89d80eae7d5fd688fac7c2c3f956de943022acd486427047eb1c584b0b58cd23"
        },
        {
          "path": "Protenix/protenix/model/modules/pairformer.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:876b29a323ebf5a9929be0b8efa6b45027108640784a250e3ce446e39ccae2eb"
        },
        {
          "path": "Protenix/protenix/model/modules/transformer.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:4d4d56b0fdf0f0fd6eff2cbcafd4a2f35b24f217746e671cac4e1238ae649ed6"
        },
        {
          "path": "Protenix/protenix/model/protenix.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:c4250bd93d592a11d9019285cfd84b71b26fa6a8aca6944a239fb31e0337e179"
        },
        {
          "path": "Protenix/protenix/model/tri_attention/forward.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:42fc170c998d3170fbf5781837a71b7d7b9f7e569be7554f39287236a7d4ae93"
        },
        {
          "path": "Protenix/protenix/model/tri_attention/op.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:e0942e6d6b06607ad399ee972465edbcb868fa1328cf39322730f3b3e4a4135b"
        },
        {
          "path": "Protenix/protenix/model/triangular/layers.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:1928d7fcb21750ce613949461ecd8ab5ff2323c19761be5b2d6f24064c3c1bbd"
        },
        {
          "path": "Protenix/protenix/model/triangular/triangular.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:4700a0abe41253a44a6504b1d11051010e5a67f9b63c29fd7535dfb033337ca2"
        },
        {
          "path": "Protenix/runner/batch_inference.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:98e2a0149359b6507e5c385665ad0365348ea8a6e361e6519c1f214801d5635b"
        },
        {
          "path": "Protenix/runner/inference.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:f2c981b38936ff863e280602fc0ce4a32bc83e3103589b0111df73c00ac297f6"
        },
        {
          "path": "Protenix/setup.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:108c7945b6ff57935285ec5ce8d2108b4be8706e09479cb04ceec67df16727de"
        },
        {
          "path": "Protenix/tests/test_attention_pair_bias.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:0702de2b5215af3bffb2c5ce09f2f999f78758c7d9ae9a58e0176ab676451de4"
        },
        {
          "path": "Protenix/tests/test_condition_transition.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:e64c300e285833444a70c23e9b7c55276821fea08876e02ba448ce27b72b3f46"
        },
        {
          "path": "Protenix/tests/test_diffusion_transformer.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:0c615b8eb2d4f819766410d7d770fc6e68c4e0602f6a2019959616b4386a4752"
        },
        {
          "path": "Protenix/tests/test_fused_dropout_add.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:c263abcbb369d3bd95cd42c2c30fd8492ca9e9c3b2a9dc92a85b549970fcd51c"
        },
        {
          "path": "Protenix/tests/test_triton_compatibility.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:c7c9bb150c3b02e01adc7c793301ea441fcc66a6e96603dbb185e11809d25d3a"
        },
        {
          "path": "README.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:eab828d1d9dbbf5adbe93d40919c1a88108a669d1d75f0179fd7c368a8711720",
          "index_digest": "sha256:eab828d1d9dbbf5adbe93d40919c1a88108a669d1d75f0179fd7c368a8711720",
          "worktree_digest": "sha256:eab828d1d9dbbf5adbe93d40919c1a88108a669d1d75f0179fd7c368a8711720",
          "untracked_digest": "absent"
        },
        {
          "path": "benchmarks/protenix/configs/mi250.yaml",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "benchmarks/protenix/profile_inference.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "docs/architecture.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:5d03fa4770bcc6c984fbc433800dd0224022756bcd6e4af8f8ea194f81c1123b",
          "index_digest": "sha256:5d03fa4770bcc6c984fbc433800dd0224022756bcd6e4af8f8ea194f81c1123b",
          "worktree_digest": "sha256:5d03fa4770bcc6c984fbc433800dd0224022756bcd6e4af8f8ea194f81c1123b",
          "untracked_digest": "absent"
        },
        {
          "path": "fireprot/adapters/protenix/runtime.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "fireprot/backends/amd/blas.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "fireprot/backends/amd/triangle_attention.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "fireprot/backends/amd/triangle_multiplication.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "fireprot/ops/registry.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "provenance/smoke_report.json",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:72bc8e52f0a2445e8602a5e07a8e436a363dcfe8907fb5b25cb2c8db770a0cf7",
          "index_digest": "sha256:72bc8e52f0a2445e8602a5e07a8e436a363dcfe8907fb5b25cb2c8db770a0cf7",
          "worktree_digest": "sha256:72bc8e52f0a2445e8602a5e07a8e436a363dcfe8907fb5b25cb2c8db770a0cf7",
          "untracked_digest": "absent"
        },
        {
          "path": "scripts/infer.sh",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:227619898fcc2bdeeecaee2a5f8d7468231dc6b2b4e0747f9a9c1c914f9d33a2",
          "index_digest": "sha256:227619898fcc2bdeeecaee2a5f8d7468231dc6b2b4e0747f9a9c1c914f9d33a2",
          "worktree_digest": "sha256:227619898fcc2bdeeecaee2a5f8d7468231dc6b2b4e0747f9a9c1c914f9d33a2",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/integration/test_protenix_numerics.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/operators/test_triangle_attention.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/operators/test_triangle_multiplication.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        }
      ]
    },
    "primary_symbols": [
      {
        "symbol": "predict",
        "file": "Protenix/runner/batch_inference.py",
        "lines": "744-930",
        "role": "CLI validation and inference orchestration"
      },
      {
        "symbol": "Protenix.forward / _main_inference_loop",
        "file": "Protenix/protenix/model/protenix.py",
        "lines": "349-630, 843-930",
        "role": "Inference stage ordering and timing boundaries"
      },
      {
        "symbol": "sample_diffusion",
        "file": "Protenix/protenix/model/generator.py",
        "lines": "119-255",
        "role": "Repeated diffusion denoiser loop"
      },
      {
        "symbol": "PairformerBlock.forward",
        "file": "Protenix/protenix/model/modules/pairformer.py",
        "lines": "103-224",
        "role": "Dominant triangle-operation and layout sequence"
      },
      {
        "symbol": "DiffusionModule.f_forward / forward",
        "file": "Protenix/protenix/model/modules/diffusion.py",
        "lines": "325-600",
        "role": "Diffusion encoder, transformer, decoder, and precision boundary"
      }
    ],
    "related_symbols": [
      {
        "symbol": "TriangleMultiplicativeUpdate.forward",
        "relationship": "called twice by PairformerBlock.forward",
        "relevance": "highest measured stable operator hotspot"
      },
      {
        "symbol": "TriangleAttention.forward",
        "relationship": "called twice by PairformerBlock.forward",
        "relevance": "second triangle hotspot and layout source"
      },
      {
        "symbol": "Attention.forward",
        "relationship": "implements attention backend dispatch",
        "relevance": "insertion point for FireProt attention backend"
      },
      {
        "symbol": "InferenceRunner.predict",
        "relationship": "called by infer_predict",
        "relevance": "autocast and model invocation boundary"
      },
      {
        "symbol": "LayerNorm",
        "relationship": "used across triangle and diffusion modules",
        "relevance": "18.72 percent of profiled aggregate kernel duration"
      }
    ],
    "execution_path": [
      "runner.batch_inference.predict validates model and kernel options and calls inference_jsons.",
      "inference_jsons constructs InferenceRunner, preprocesses each JSON, and enters infer_predict.",
      "InferenceRunner.predict moves tensors, selects autocast, and calls Protenix.forward in inference mode.",
      "Protenix.forward enters main_inference_loop and _main_inference_loop.",
      "get_pairformer_output performs input embedding, four Mini recycling cycles, template/MSA work, and the Pairformer stack.",
      "sample_diffusion iterates the noise schedule and calls the denoiser once per diffusion step.",
      "The confidence and distogram heads consume trunk and sampled coordinates before the runner serializes outputs."
    ],
    "pdg_constraints": [
      {
        "description": "Pair representation z must preserve the native semantic order through both inplace-safe and residual paths.",
        "affected_statements": [
          "Protenix/protenix/model/modules/pairformer.py:139",
          "Protenix/protenix/model/modules/pairformer.py:171",
          "Protenix/protenix/model/modules/pairformer.py:218"
        ],
        "implementation_consequence": "Dispatch may replace an individual triangle operation but must preserve outgoing multiplication, incoming multiplication, start attention, transpose, end attention, transpose, and pair transition order."
      },
      {
        "description": "Diffusion token state is explicitly upcast to FP32 before the transformer and then consumed by the atom decoder.",
        "affected_statements": [
          "Protenix/protenix/model/modules/diffusion.py:454",
          "Protenix/protenix/model/modules/diffusion.py:470",
          "Protenix/protenix/model/modules/diffusion.py:483"
        ],
        "implementation_consequence": "Do not silently lower diffusion-transformer precision; any mixed-precision path needs separate scientific validation."
      },
      {
        "description": "Inference and training take different Pairformer residual paths.",
        "affected_statements": [
          "Protenix/protenix/model/modules/pairformer.py:139",
          "Protenix/protenix/model/modules/pairformer.py:171"
        ],
        "implementation_consequence": "Initial custom kernels are inference-only and must fall back for autograd or unsupported inplace semantics."
      }
    ],
    "architectural_patterns": [
      {
        "pattern": "Model adapters own model-specific tensor and call-site mapping.",
        "example_location": "docs/architecture.md",
        "usage_guidance": "Keep Protenix field names and semantics inside fireprot/adapters/protenix."
      },
      {
        "pattern": "Operators expose semantic contracts while AMD backends own implementations.",
        "example_location": "docs/architecture.md",
        "usage_guidance": "Place registry and contracts under fireprot/ops and hardware implementations under fireprot/backends/amd."
      },
      {
        "pattern": "Native implementation remains the reference and fallback.",
        "example_location": "Protenix/protenix/model/triangular/triangular.py",
        "usage_guidance": "Capability checks must fail closed to the existing torch path."
      }
    ],
    "files_to_modify": [
      {
        "file": "benchmarks/protenix/profile_inference.py",
        "symbols": [],
        "intended_change": "Add reproducible cold/warm stage timing, torch-profiler shape capture, rocprof import, and structured result emission."
      },
      {
        "file": "benchmarks/protenix/configs/mi250.yaml",
        "symbols": [],
        "intended_change": "Declare the initial length, cycle, step, sample, dtype, seed, cache, and repetition matrix."
      },
      {
        "file": "fireprot/ops/registry.py",
        "symbols": [],
        "intended_change": "Add model-neutral operation signatures, capability predicates, priority, cached selection, and native fallback."
      },
      {
        "file": "fireprot/backends/amd/blas.py",
        "symbols": [],
        "intended_change": "Add explicit process-start BLAS policy and offline per-signature benchmark records."
      },
      {
        "file": "fireprot/backends/amd/triangle_multiplication.py",
        "symbols": [],
        "intended_change": "Register AMD FP32 inference implementations for outgoing and incoming triangle multiplication."
      },
      {
        "file": "fireprot/backends/amd/triangle_attention.py",
        "symbols": [],
        "intended_change": "Register an online-softmax AMD FP32 inference implementation with mask and bias semantics."
      },
      {
        "file": "fireprot/adapters/protenix/runtime.py",
        "symbols": [],
        "intended_change": "Install FireProt operators before Protenix model construction and record the selected implementation manifest."
      },
      {
        "file": "Protenix/runner/batch_inference.py",
        "symbols": [
          "predict"
        ],
        "intended_change": "Accept the FireProt backend option without changing native defaults."
      },
      {
        "file": "Protenix/protenix/model/triangular/triangular.py",
        "symbols": [
          "TriangleMultiplicativeUpdate.forward",
          "TriangleAttention.forward"
        ],
        "intended_change": "Add capability-gated FireProt dispatch with the native torch path as fallback."
      },
      {
        "file": "Protenix/protenix/model/triangular/layers.py",
        "symbols": [
          "Attention.forward"
        ],
        "intended_change": "Add the FireProt attention dispatch point after projections and before the native logits materialization."
      },
      {
        "file": "scripts/infer.sh",
        "symbols": [],
        "intended_change": "Route optional accelerated runs through the Protenix adapter and expose explicit backend controls."
      },
      {
        "file": "tests/operators/test_triangle_multiplication.py",
        "symbols": [],
        "intended_change": "Add numerical, shape, mask, layout, fallback, and aliasing tests."
      },
      {
        "file": "tests/operators/test_triangle_attention.py",
        "symbols": [],
        "intended_change": "Add numerical, softmax, bias, mask, layout, and fallback tests."
      },
      {
        "file": "tests/integration/test_protenix_numerics.py",
        "symbols": [],
        "intended_change": "Add same-seed native-versus-accelerated structure and confidence comparisons."
      }
    ],
    "tests": [
      {
        "file": "Protenix/tests/test_attention_pair_bias.py",
        "scenarios": [
          "Retain dense and local-attention shape coverage under native defaults.",
          "Add a FireProt option check only if the upstream test can run without the top-level package; otherwise keep integration coverage top-level."
        ]
      },
      {
        "file": "Protenix/tests/test_triton_compatibility.py",
        "scenarios": [
          "gfx90a unsupported or import failure -> native torch fallback with no model-load failure."
        ]
      },
      {
        "file": "tests/operators/test_triangle_multiplication.py",
        "scenarios": [
          "N in {1,16,31,32,62,124,127,128,256}, both directions, dense and sparse masks -> close to native FP32.",
          "Noncontiguous input, unsupported dtype, training/autograd, or unsupported CK problem -> deterministic native fallback.",
          "inplace-safe enabled -> output is correct and documented aliasing constraints hold."
        ]
      },
      {
        "file": "tests/operators/test_triangle_attention.py",
        "scenarios": [
          "Start and end orientation, optional mask, both bias tensors, ragged lengths around tile boundaries -> close to native FP32.",
          "All-masked rows and extreme logits -> finite results matching native semantics.",
          "Unsupported architecture, dtype, head dimension, or layout -> native fallback."
        ]
      },
      {
        "file": "tests/integration/test_protenix_numerics.py",
        "scenarios": [
          "Same model/input/seed/C4/S1 and C4/S5 -> accelerated FP32 coordinates and numeric confidence stay within locked tolerances.",
          "All eight development cases -> no finite-output regression; coordinate metrics use only coordinates[mask].",
          "Fallback-only run -> output schema and native behavior remain unchanged."
        ]
      }
    ],
    "verification_commands": [
      "cd Protenix && PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_attention_pair_bias.py tests/test_condition_transition.py tests/test_diffusion_transformer.py tests/test_fused_dropout_add.py tests/test_triton_compatibility.py",
      "PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/operators/test_triangle_multiplication.py tests/operators/test_triangle_attention.py tests/integration/test_protenix_numerics.py",
      "python benchmarks/protenix/profile_inference.py --config benchmarks/protenix/configs/mi250.yaml --backend native",
      "python benchmarks/protenix/profile_inference.py --config benchmarks/protenix/configs/mi250.yaml --backend fireprot",
      "rocprofv3 --stats -- python benchmarks/protenix/profile_inference.py --config benchmarks/protenix/configs/mi250.yaml --backend fireprot --profile-one"
    ],
    "risks": [
      "Per-call switching of PyTorch's process-global BLAS preference is unsafe; global selection is process-start only and custom operations own finer dispatch.",
      "Numerically equivalent reduction reordering can perturb diffusion trajectories; end-to-end same-seed gates are mandatory.",
      "CK does not support every observed BMM signature; capability checks and fallback are part of correctness.",
      "Static GitNexus impact did not resolve dynamic nn.Module callers, so source call-site search remains authoritative.",
      "The current measurements cover two 124-token development cases and cannot establish long-sequence or Base-model speedups."
    ],
    "assumptions": [
      "Before implementation benchmarking, verify DiamondHill runtime_manifest, Protenix commit d3b4db6, weights, ROCm/PyTorch build, GPU clock state, and exclusive device use.",
      "Before enabling the existing Triton triangle-attention kernel, rerun its correctness and performance tests on gfx90a; its presence in source is not compatibility evidence.",
      "Before locking numerical thresholds, repeat the native same-seed run to establish measurement and serialization noise."
    ],
    "open_questions": [
      "How do hotspot shares change for Base weights, long sequences, ligand-heavy inputs, and the production diffusion-step count?",
      "Does AMD-compatible Triton or a CK/HIP implementation win for triangle attention after compilation and layout costs?",
      "Which observed shapes justify CK once unsupported problems, workspace, and first-use compilation are included?",
      "What independent validation set will be used after the eight development cases?"
    ],
    "avoid": [
      "Do not repeat full repository discovery.",
      "Do not present the session's one-warm-sample comparison as a benchmark result.",
      "Do not enable BF16 by default from the current evidence.",
      "Do not change diffusion-transformer FP32 upcast semantics without an independent accuracy study.",
      "Do not add Protenix as a top-level gitlink or submodule; commit nested source changes in the Protenix checkout.",
      "Do not version runtime weights, CCD resources, outputs, or autotune caches."
    ]
  }
}
~~~

## 12. Assumptions and Open Questions

### Assumptions to re-verify

- [assumed] DiamondHill runtime assets still match provenance/runtime_manifest.json, and the Protenix checkout remains at d3b4db6 before implementation measurements. Verify commit, weights, CCD paths, ROCm/PyTorch versions, clocks, and device exclusivity in every run manifest.
- [assumed] PyTorch's reported hipBLASLt/hipBLAS/CK preferences still map to the installed ROCm build. Verify with a startup self-test rather than relying on backend names alone.
- [assumed] The existing Triton triangle-attention algorithm can compile on gfx90a. Treat compilation, correctness, and speed as separate gates.
- [assumed] The initial FP32 end-to-end tolerances are strict enough to detect meaningful drift and loose enough for legal reduction reordering. Establish native rerun noise before locking them.

### Open questions

1. How do hotspot shares change for Base weights, long sequences, ligand-heavy inputs, and the production diffusion-step count?
2. Does AMD-compatible Triton or a CK/HIP implementation win triangle attention after compile time, workspace, and layout costs?
3. Which signatures justify CK after unsupported-problem fallback and first-use cost?
4. What independent validation set will follow the eight development cases?
5. Does diffusion dominate at the production step count enough to prioritize AtomTransformer attention or conditioned transitions after Pairformer work?

### Explicitly deferred follow-ups

- BF16/FP16 promotion and mixed-precision diffusion.
- Diffusion transformer and local atom-attention kernels until S-production profiles exist.
- HIP graph capture until shapes, allocations, and random-state behavior are stable.
- Multi-GPU inference and distributed scheduling.
- AtlasFold, OpenFold, and ESMFold adapters; this work defines reusable contracts but implements only Protenix integration.

## 13. Definition of Done

- The benchmark harness emits raw, reproducible cold/warm measurements with every required model, input, dtype, cycle, step, sample, seed, cache, software, hardware, and backend field.
- Native measurements are repeated on MI250 and replace all ephemeral planning numbers in documentation.
- FireProt registry selection is deterministic, capability-gated, versioned, reported, and falls back to native behavior.
- Triangle multiplication and triangle attention pass the full operator scenario matrix in FP32.
- Unsupported architectures, dtypes, layouts, masks, training/autograd calls, and CK problems reach the native path without model-load or output-schema failures.
- Same-seed integration checks pass locked coordinate and confidence tolerances, and all eight development cases retain finite outputs.
- Every promoted signature passes its operator performance gate; every advertised end-to-end bucket passes the repeated median/p95 gate without peak-memory regression.
- Reprofiling demonstrates the changed kernel mix, launch/allocation/transfer counts, and identifies remaining hotspots.
- README and architecture documentation describe only implemented behavior and reproduced measurements.
- Top-level FireProt changes and nested Protenix changes are committed in their respective repositories; no weights, outputs, runtime cache, Git metadata, or Protenix gitlink is added to top-level Git.
