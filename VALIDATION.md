# Recorded verification

The project was verified on Isaac-PC before this GitHub snapshot. Publishing the
repository does not rerun the experiment or change its algorithms.

- 33 focused sampler-contract tests passed during desktop setup.
- 10 HTTP/control tests passed for the immediate-input, replay, reset and error
  behavior, including the transient Windows status-file read case.
- UI logic checks passed for fixed input, one grouped event per anchor, exact
  slide RGB roles, overlapping/self memberships, duplicate counts, all 451 frames
  in order, editable cadence, reset/restart, and full-length lines without tips.
- Real baseline/36, modified/36 and baseline/100 runs at seed 42 matched the
  corresponding earlier desktop final coordinates exactly. Pair arrays matched
  fresh untouched sampler calls; streamed frames matched the full real archive.
- Later browser QA verified whole graph sections at 760×600, 1000×700 and
  1440×900, with equal-scale proportions, unclipped axes and stacked sections.
  At the smallest size, input/output section heights were about 507/521 pixels.
- The original 60 matched trials, all datasets, static research plots, pinned
  baseline and separate sampler-only modification are preserved.

Original cloud checks and limitations remain in `experiment/REPORT.md` and its
verification artifacts. They are historical records, separate from later desktop
checks. Machine-local logs, browser screenshots, temporary epoch replays and
desktop backups remain on the original machine and are intentionally not tracked.

Use the non-training checks in the root README to verify a new checkout.
