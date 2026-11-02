# ParamRepulsor: exact sampler comparison

## Result first

The requested sampler rules are enforced. On these three synthetic datasets, compare cluster separation **and** local-neighbor fidelity rather than calling the change universally better. Both variants already preserve the supplied cluster labels very well; tight-looking clusters alone are not proof of better fine structure.

| Points | Measure | Baseline mean ± SD | Modified mean ± SD | Paired change (modified − baseline) |
|---:|---|---:|---:|---:|
| 36 | Silhouette ↑ | 0.8799 ± 0.0179 | 0.9700 ± 0.0029 | +0.0901 |
| 36 | Original 10-NN recall ↑ | 0.9672 ± 0.0157 | 0.9314 ± 0.0100 | -0.0358 |
| 36 | Distance-rank correlation ↑ | 0.8837 ± 0.1173 | 0.9235 ± 0.0113 | +0.0398 |
| 50 | Silhouette ↑ | 0.8480 ± 0.0162 | 0.9286 ± 0.0206 | +0.0806 |
| 50 | Original 10-NN recall ↑ | 0.8960 ± 0.0224 | 0.8498 ± 0.0093 | -0.0462 |
| 50 | Distance-rank correlation ↑ | 0.9511 ± 0.0363 | 0.9037 ± 0.0209 | -0.0474 |
| 100 | Silhouette ↑ | 0.8712 ± 0.0278 | 0.9095 ± 0.0350 | +0.0383 |
| 100 | Original 10-NN recall ↑ | 0.8736 ± 0.0187 | 0.8627 ± 0.0186 | -0.0109 |
| 100 | Distance-rank correlation ↑ | 0.8557 ± 0.0873 | 0.8367 ± 0.0715 | -0.0191 |

Values summarize all 10 predeclared matched seeds (0–9) per dataset. SD is between-seed sample SD, not a confidence interval. All 60 outcomes are retained. This is a diagnostic 2D→2D sampling stress test, **not** a reproduction of the paper's high-dimensional benchmarks or proof of better general-purpose dimensionality reduction. Three nested datasets are not three independent real-world replications. Both sampler changes are combined, so this is not an ablation of their individual effects.

## What was actually run

- Official source: https://github.com/hyhuang00/ParamRepulsor
- Untouched baseline commit: `be8df72b1ac9041be3aae3d99f16f0d392b492dc`.
- Paper: https://arxiv.org/abs/2411.15894 (the inspected PDF is v1).
- Exact existing input ZIP: Library `libfile_59b61f1348648191b85bb99fdd49d9cb`, SHA-256 `f3b9fb14b16eea8a2d0ad38ddc61cc6b61fc157d849f51c8768fb12ee5b6a652`.
- Actual CSV cluster sizes: 36 = 11/12/13; 50 = 11/14/25; 100 = 11/11/38/40. They are nested with stable IDs and coordinates; 100 adds a fourth group, so size is not the only changing variable.
- Features: only `math_score`, `english_score`, in that order, float32. Group and point ID are used only for evaluation/graphics.
- Defaults unchanged: 2 outputs, 10 NN / 5 MN / 20 FP, 450 epochs, three 100-unit hidden layers with default SiLU activation, Adam LR0.001, batch size1024, no feature scaling, original loss and weight/constant schedules. CPU float32; deterministic Torch; one CPU/Numba thread per isolated trial. The actual dataset size is below one batch.
- For every one of 30 matched pairs, exact equality is checked for input rows/IDs/labels, NN partners, initial weights, initial embedding and initial training RNG state.
- Pair generation happens **once before training**, not freshly in each of 450 epochs in this pinned implementation. These experiments test the released code, not an alternative interpretation of the paper pseudocode.

## Exact source change

Only `src/parampacmap/utils/data.py` differs. `minimal.diff` applies to the pinned untouched baseline. No model, loss, NN selection, schedules or training code changed.

1. MN excludes the anchor and its existing NN set.
2. Each trial draws six distinct eligible points and selects the second closest under the unchanged original metric/tie rule.
3. Previously accepted MN partners remain eligible in the candidate pool. A repeated selected partner causes rejection of the **whole six-point draw**, followed by a fresh draw. Retry does not reseed the deterministic stream each time.
4. FP excludes the anchor and NN set and chooses distinct partners without replacement.
5. MN/FP overlap remains allowed.
6. Seeded and nonseeded routes share these rules.

The unmodified code was inspected and exercised: its MN candidates use replacement and no self/NN exclusion; final MN duplicates are unchecked. The FP helper's duplicate check does not control acceptance, and the wrapper also redraws a full vector independently for each final partner. NN/FP exclusion itself already works. Self-pairs do not literally separate a point from itself: their coordinate-difference gradient is zero; they consume/reweight loss terms. Sampling anomalies do not by themselves prove worse embeddings.

### Measured violations, summed over all 10 seeds

Duplicate excess = requested partner count minus distinct partner count for an anchor, summed across anchors. NN overlap counts sampled partners lying in that anchor's NN set (not just distinct overlaps). All counts are from saved actual training pairs.

| Points | Variant | MN self | MN/NN overlap | MN duplicate excess | FP self | FP duplicate excess | MN/FP overlap (allowed) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 36 | upstream | 19 | 1100 | 171 | 293 | 2173 | 405 |
| 36 | modified | 0 | 0 | 0 | 0 | 0 | 1448 |
| 50 | upstream | 10 | 1091 | 185 | 255 | 2058 | 545 |
| 50 | modified | 0 | 0 | 0 | 0 | 0 | 1280 |
| 100 | upstream | 5 | 692 | 166 | 204 | 1912 | 874 |
| 100 | modified | 0 | 0 | 0 | 0 | 0 | 1146 |

### Feasibility and bounded retries

With defaults and MN/FP overlap allowed, FP exclusions set a minimum of **31 points**, not36. At least six eligible MN candidates are needed. With `m` eligible distinct-distance points, only ranks2 through `m−4` can be second of six, so at most `m−5` distinct final MN outcomes are possible. The patch checks cardinalities and exact tie-aware support near that boundary. It explicitly raises on infeasible requests.

Even a feasible request can require arbitrarily many random retries. There is a documented cap of **100,000 whole draws per final MN**; reaching it raises an error, with no distribution-changing fallback. None of these dataset runs hit it. Candidate-pool materialization and without-replacement sampling have unmeasured large-N cost; do not claim this implementation has negligible production overhead.

## Graphics and interpretation

All graphics use Matplotlib, never image generation. Group colors and point IDs are consistent.

- `plots/comparison_overview_seed0.png`: all three inputs and raw baseline/modified outputs.
- `plots/input_output_{36,50,100}_seed0.png`: readable per-dataset inputs and raw outputs with labeled axes; point-ID overlays are in companion SVGs.
- `plots/movement_aligned_{36,50,100}_seed0.png`: faint original points → arrows → strong final points.
- `plots/all_seeds_metrics.png`: every matched seed for six metrics, not just the illustration.
- Clean SVG versions and optional `*_point_ids.svg` overlays are included. Dense clusters can make individual ID labels overlap; IDs and exact rows are also retained in every NPZ/CSV.

Seed0 was fixed before seeing outcomes; it is not the best-looking seed. Raw embeddings have arbitrary translation, rotation/reflection and scale and do not use original score units. For correspondence arrows only, each output receives a single least-squares global similarity fit to the input (translation + orthogonal rotation/reflection + uniform scale). Exact transforms and whether reflection was used are saved in each trial JSON. This removes arbitrary coordinate-system differences, not local deformation. Arrows are **input-to-final correspondences, not physical motion or actual optimization trajectories**. Actual raw epoch embeddings are separately retained in NPZ files and the simulator.

## Real-training Start-button simulator

`simulator/` is a local, localhost-only app. Start launches the real selected model for the selected dataset/seed using the unchanged training loop and streams actual epoch coordinates. It is not a force simulation or interpolated path. It supports baseline/modified, dataset36/50/100, real progress, stable identities and repeat runs. See its README for access and verification details. No public hosting or paid compute was used.

## Verification and runtime caveats

- 33 focused sampler-contract tests pass, including seeded/nonseeded paths, duplicate retry semantics, retained MN eligibility, allowed MN/FP overlap, all supported metrics, infeasibility, forced retry failure and exhaustive tie support.
- The cloud sandbox denies Unix-socket IPC used by PyTorch inference workers. A **separate harness-only inference transport adapter** runs the identical single-batch model inference in process, identically for both arms. The baseline source itself is untouched.
- Observer snapshots are direct, read-only model forwards and assert unchanged Torch RNG. On the 36-point seed0 integration check, observed training and no-observer training produce bitwise-identical final outputs and NN/MN/FP arrays. Original inference arithmetic (same dataset/order/batch, in-process transport) also matches exactly. See `results/harness_verification.log`.
- Raw `seconds` values are **not fair sampler benchmarks**: baseline typed Numba functions compile during import before the fit timer, while modified lazy functions compile during first fit; trial workers also contend concurrently. They are retained for transparency only. Warmed, sequential pair-generation timings (including Annoy NN construction) are in results/warmed_pair_benchmarks.csv, measured after all training stopped and after one warmup per size/variant. Median seconds: N36: baseline 0.00514, modified 0.00385; N50: baseline 0.00760, modified 0.00601; N100: baseline 0.01734, modified 0.01545. These are tiny-data CPU measurements, not production scalability evidence.
- The pristine upstream runtime suite was blocked by that IPC restriction, not reported as passed. The unchanged official suite passed7/7 against baseline and7/7 against modified with a separately labeled test-only inference transport adapter (see results/upstream_suite_*_adapted.log). These are not pristine-environment passes.
- Full browser visual QA of the simulator is blocked by Chromium socket EPERM in this cloud environment; HTTP/backend/JavaScript checks are documented separately. Static plots were rendered and visually inspected.

## Reproduce

Use Python3.11 (upstream supports >=3.9,<3.12). From the extracted project root:

```bash
bash experiment/setup.sh
.venv/bin/python -m pytest experiment/tests -q
.venv/bin/python experiment/run_all.py
MPLCONFIGDIR=/tmp/paramrep-mpl .venv/bin/python experiment/analyze_plot.py
.venv/bin/python experiment/verify_harness.py
.venv/bin/python experiment/simulator/server.py
```

Open the localhost URL printed by the simulator in the same computer's browser. Training can be faster than the browser poll rate; it may skip displayed epochs but saves all actual epoch frames. `requirements-lock.txt` records the actual environment; `requirements-runtime.txt` is the portable pinned dependency list excluding the local-source package and separately installed CPU Torch wheel.

## Files

- `upstream/`: untouched official source and Apache2.0 license; `experiment/upstream.bundle` preserves the pinned Git commit.
- `modified/`: separate sampler-only source copy.
- `inputs/`: original ZIP and exact extracted datasets/manifest.
- `experiment/results/`: all60 trial JSON/NPZ pairs, all-trial and paired-difference CSVs, matching checks, verification logs.
- `experiment/plots/`: PNG/SVG graphics.
- `experiment/minimal.diff`, sampler tests/review, runnable experiment and simulator scripts.

Citation: Huang, Haiyang; Wang, Yingfan; Rudin, Cynthia. *Navigating the Effect of Parametrization for Dimensionality Reduction*. NeurIPS2024.
