# Sampler-only patch review

Pinned official source: `hyhuang00/ParamRepulsor`, commit `be8df72b1ac9041be3aae3d99f16f0d392b492dc`.
Only `src/parampacmap/utils/data.py` differs between baseline and modified trees.
`minimal.diff` applies cleanly to that baseline. Baseline tracked files remain untouched.

## Verified baseline behavior

The released MN sampler uses six independent `randint` draws with replacement,
without excluding the anchor or NN partners, and does not check repeated final MNs.
The released FP helper's duplicate check does not control acceptance: its acceptance
`else` belongs to the separate NN-exclusion loop. Moreover, the FP pair wrappers
draw an independent full FP vector for every final output column, so fixing only
the helper would not make the final partner list distinct. Anchor self-pairs are
not excluded in the FP helper call.

Concrete reproduction on the unchanged supplied 36-point CSV, seed 42:

| Measure | MN | FP |
| --- | ---: | ---: |
| Anchor rows containing self | 1 | 21 |
| Duplicate excess (requested count minus unique count, summed) | 14 | 220 |
| Anchor rows containing duplicates | 12 | 36 |
| Unique overlap with NN partners, summed across anchors | 92 | 0 |

These verify sampling properties, not an embedding-quality improvement.

## Exact patched rules

- NN selection code is unchanged.
- Each anchor's MN candidate pool excludes the anchor and its existing NN set.
- Every MN draw samples six distinct points uniformly without replacement from
  that pool, then uses the original metric and the original first-occurrence tie
  handling to choose the second closest.
- Previously accepted final MN partners remain in the candidate pool. If a draw
  selects one again, the entire draw is discarded and six candidates are drawn
  again. Seeding occurs once per requested final partner, outside the retry loop.
- FP excludes the anchor and existing NN set. One without-replacement draw per
  anchor supplies all requested FP partners.
- MN/FP overlap is deliberately allowed.
- Seeded and nonseeded public entry points use the same core implementations.
  Original deterministic per-partner MN seed offsets are preserved. FP's old
  per-column reseeding is replaced with one anchor-specific seed because the
  final FP list is now a single without-replacement draw.

## Infeasibility and retry limit

Counts must be nonnegative integers, and NN count must be less than dataset size.
FP counts greater than the available candidate count fail explicitly. MN needs at
least six candidates when any MN is requested.

For `m` eligible points at strictly distinct distances, only ranks 2 through
`m-4` can become the second of a six-point draw: `m-5` possible final partners.
Ties can enlarge this support because sampled order breaks ties. Near the
cardinality edge, the patch computes exact tie-aware support and rejects a
request exceeding it. For defaults (5 MN, 10 NN, 20 FP), all three supplied
datasets have ample support; allowing MN/FP overlap means 31, not 36, is the
cardinality minimum set by FP exclusions.

An extremely skewed but feasible request can still take arbitrarily many random
draws. The implementation caps retries at **100,000 full six-point draws per
requested final MN**, then raises `RuntimeError` with no fallback. This prevents
unbounded hanging; it does not silently remove previous MN candidates or change
the requested distribution. Very rare valid requests could therefore fail
explicitly. The retry-failure test deliberately forces this exception; no ordinary dataset contract test required a fallback or hit the retry cap.

## Focused tests

`experiment/tests/test_sampler.py` checks actual datasets and seeded/nonseeded
paths; exact seeded reproducibility; unchanged NN partners; six-distinct draws;
second closest for all four supported metrics; scripted whole-draw duplicate
rejection while retaining prior MN candidates; allowed MN/FP overlap; infeasible
counts; tie-expanded support; exhaustive tie-aware support; and explicit retry
failure without fallback.

### Runtime test caveat

The pristine official test suite was started against modified source but could
not complete its first inference in this cloud sandbox: default PyTorch worker
transport requires unavailable Unix IPC. It was interrupted (exit 130), with no
assertions reported. This is **not** a passed official-runtime suite.
`inprocess_inference_adapter.py` is a separately named pytest-only transport
adapter. It preserves inference dataset, order, batching and arithmetic while
using an in-process loader. Official test files and source trees are unchanged.
Adapted suite outcomes are retained separately in `results/*_adapted.log`.
Baseline adapted suite: **7 passed in 77.00 seconds**.
Modified adapted suite: **7 passed in 97.63 seconds**.
These include the unchanged official 450-epoch, 1000-point pair-saving test.
Focused sampler contract verification: **33 distinct cases passed** (32-case run
plus the added exhaustive tie-support test, with unchanged sampler source).

### Timing caveat

The pinned baseline uses eager typed Numba decorators, so JIT compilation occurs
during import. The modified shared implementations compile lazily on first use.
Raw first-fit timing therefore includes different compilation costs and must not
be presented as a fair steady-state runtime comparison. Warm both implementations
before measuring sampling throughput, or explicitly label cold-start timing.
