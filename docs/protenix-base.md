# Protenix Base integration

Base-default v0.5.0 uses the existing Rust request and supervised Protenix worker
boundary. Its checkpoint passes strict CPU loading: 368,088,191 parameters,
4,092 FP32 tensors, no missing or unexpected keys. See
[checkpoint evidence](../provenance/protenix_base_checkpoint.json).

## Initial workload

The qualified N600 request and prepared N1000/N1536 requests select FP32, C10/D200/S1, seed 101,
sequence-only monomers without MSA or templates, on HIP device 7. Base's upstream
default sample count is **five**; these single-sample experiments are not the full
upstream default workload. Keep all scientific settings identical within each
native/FireProt comparison. Mini C4/D1 speedups do not establish Base performance.

```bash
target/release/fireprot inspect configs/requests/protenix-base-n600-native.json
target/release/fireprot infer configs/requests/protenix-base-n600-native.json \
  configs/diamondhill/protenix-base-init-only.json
target/release/fireprot infer configs/requests/protenix-base-n600-fireprot.json \
  configs/diamondhill/protenix-base-init-only.json
```

Copy requests with fresh run IDs and output paths before repeating them. The
N1536 equivalents use `protenix-base-n1536-{native,fireprot}.json`.
The same profile can be used for both arms: the native backend does not install
FireProt hooks. Only HIP devices 6 and 7 are authorized for these experiments.

## Transplant boundaries

Existing HIP/CK triangle multiplication, triangle attention, LayerNorm, transition
and layout paths retain their exact shape/dtype/stride/device capability guards.
Unsupported signatures retain native execution. Their selection on Base must be
verified in `operator-runtime.json`, and model outputs must pass the existing
Protenix coordinate/confidence gates before reporting performance gains.

Fast model construction and skipped template preparation remain disabled in the
original broad-operator Base profile. The separate construction experiment qualifies
Base initialization; template omission still only qualifies Mini. Base's deeper stack
and full diffusion workload require independent output validation. The first GPU transplant failed the coordinate gate; no accepted Base GPU-operator speedup
is claimed. See the result below.


## First N1000 transplant: rejected

The native and candidate runs both load and produce accepted finite output
bundles, but the paired correctness gate rejects the candidate. Full invocation
times are 419.053 s and 185.192 s; model times are 349.312 s and 115.087 s.
These are diagnostic timings, **not an accepted speedup**. Aligned coordinate
RMSD is 0.0467943 Angstrom, above the unchanged 0.001 gate. Summary confidence
maximum delta is 0.000518799. Atom metadata and CIF atom identities match.
CK attention and HIP packed triangle multiplication each execute 1,048 times.
[Evidence](../provenance/protenix_base_first_transplant.json).

The identical native repeat also failed: aligned RMSD 0.00828346 Angstrom.
The first measurements use upstream nondeterministic execution, so they cannot
isolate the cause of the candidate difference. The maintained Base worker now
requires strict deterministic algorithms and uninitialized-memory filling in
both backends and checks this again at the model boundary. Per the updated experimental scope, use approximately 600 residues
for focused diagnostics and N1000 for E2E qualification. N1536 examples are
prepared but are not required or being run in this round.


## Fast construction qualification

The CPU qualification script in `experiments/protenix_base/check_initialization.py`
compares full Base construction, strict checkpoint loading, all state-dict tensors,
all named buffers, and Python/NumPy/Torch CPU random states. It passes; construction
alone takes 50.222 s natively versus 3.635 s with deferred initializers. This is not
an E2E measurement. The existing complete-checkpoint check and native recovery
remain mandatory. [CPU evidence](../provenance/protenix_base_cpu_initialization.json).

The strict N600 experiment isolates this optimization, disables GPU operator
substitutions, and retains C10/D200/S1. All nine arrays are bitwise equal;
summary confidence and raw coordinate differences are zero. CIF atom identities
match. Complete invocation time is **179.350 -> 127.047 s (1.412x)**, from one
serial pair with fresh processes and unflushed filesystem caches. This is not a
three-pair timing trend or a steady-state model acceleration claim. The runtime
confirms fast construction selected once, no recovery, and all 1,048 triangle
attention/multiplication calls in each family using native implementations.
[Full result](../provenance/protenix_base_strict600_initialization.json).

Use the qualified initialization-only profile for this scope:

```bash
target/release/fireprot infer configs/requests/protenix-base-n600-native.json \
  configs/diamondhill/protenix-base-init-only.json
target/release/fireprot infer configs/requests/protenix-base-n600-fireprot.json \
  configs/diamondhill/protenix-base-init-only.json
```

The input is an explicitly documented 600-residue prefix of the N1000 development
sequence. It is a performance/repeatability diagnostic, not an experimental
structure-quality benchmark. N1000 strict operator qualification remains open;
N1536 is outside this round's requested scope.

## Strict N1000 attention isolation

A subsequent strict deterministic N1000 C10/D200/S1 comparison enables only CK
attention and qualified initialization, leaving triangle multiplication and the
other GPU substitutions native. CK attention executes 1,048 times, but the
candidate still fails: aligned RMSD 0.0749593 Angstrom and summary confidence
maximum delta 0.00131607. Diagnostic invocation times are 430.648 s and
216.682 s; **no performance verdict is accepted**.
[Evidence](../provenance/protenix_base_strict1000_attention.json).
An initialization-only N1000 control is in progress to check the reference at
this length independently of CK.

An experimental HIP bias/softmax kernel retaining native QK/PV arithmetic passes
30 synthetic bitwise cases at N600/N1000. This establishes only an operator screen;
trained-attention replay and complete model qualification remain outstanding.
[Screen evidence](../provenance/protenix_base_bias_softmax_screen.json).
