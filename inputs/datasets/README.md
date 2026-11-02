# ParamRepulsor: 2D Dataset Files

Three **synthetic**, pre-made point datasets to evaluate whether sampling-rule changes affect ParamRepulsor's ability to preserve an obvious 2D grouped structure. No dimensionality reduction is required for this test (2D input -> 2D output); it tests neighborhood preservation and optimization behavior.

## Files

- `paramrep_2d_36.csv`: 36 points, groups of 11 (High Both), 12 (Math Strong), 13 (English Strong).
- `paramrep_2d_50.csv`: 50 points, groups of 11 (High Both), 14 (Math Strong), 25 (English Strong).
- `paramrep_2d_100.csv`: 100 points, groups of 11 (High Both), 11 (Low Both), 38 (Math Strong), 40 (English Strong).
- `manifest.json`: exact cluster centers, generation seed, verification counts.

**Nested selection:** Every point in 36 is also in 50, and every point in 50 is also in 100. Their scores and IDs are unchanged. This reuses the same points as dataset size increases. One tradeoff: 100 introduces a fourth cluster (Low Both), so group count is not held constant across all three sizes.

## CSV columns

- `point_id`: stable identifier, e.g. P023.
- `math_score`: X coordinate, numerical value on 0–100 scale.
- `english_score`: Y coordinate, numerical value on 0–100 scale.
- `group`: *known label for plotting/evaluation only*. **Do not include group or point_id as input features to ParamRepulsor.**

Feed only the two numerical score columns to the algorithm. Color points on scatterplots using the `group` column.

## Dataset design and validations

The points sit in tight, visibly separate clusters:

- High Both: around (88, 88)
- Low Both: around (49, 49)
- Math Strong: around (91, 55)
- English Strong: around (55, 91)

Each point is a different 2D location. The maximum distance from each cluster center is 5.3 score units. All 10 **plain Euclidean** nearest neighbors of every point are inside the same known cluster in **all three datasets** (verified). The released ParamRepulsor code uses local-distance scaling to rank nearest-neighbor candidates, so the verification is about unscaled Euclidean distances, not a guarantee about the approximate Annoy+scaled implementation in every run.

For a fair original-vs-modified comparison: reuse the exact datasets and neural-network settings, use repeated seeds, and report counts of self-pairs, NN/MN overlap, duplicate samples, and timing alongside the output plots. Label the outputs as experiments, not improvements, until the results are measured.

**Minimum-size caveat:** At 36 points, you can have 10 NN, 5 MN, and 20 FP that are all distinct from the anchor and each other. But if you additionally require FP points to lie farther than the chosen MN points, valid sets may not exist; use an explicit fallback or keep that rule in a separate experiment.

## Simple Python CSV loading

The files can be loaded with any CSV reader. For numerical code, select the `math_score` and `english_score` columns in that order as your input matrix. The group labels are only for evaluation.

## Reproduction

Seed: `4612026`. Synthetic clustered score pairs were generated with independent, uniform-in-disk jitters of radius at most 5.3 around the centers listed above. Selection of smaller files retains fixed point IDs, coordinates, and groups from the full 100-point file.
