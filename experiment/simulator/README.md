# Actual sampler and training demo (local only)

Double-click `Start Simulator.bat` in the Windows project root and manually open
**http://127.0.0.1:8765**. For this presentation-only update, use Ctrl+F5; no
server restart is required. The existing launcher remains unchanged.
The released baseline is the talk default; modified, 36/50/100 and seed controls
remain available. No algorithms, counts, model, loss or optimizer are changed.

## Immediate input and real pairs

All exact CSV datasets are preloaded in the page as original float32 score
coordinates. Input appears synchronously before network status, Torch imports,
compilation, pair selection or training. Its geometry, identities and group
colors stay fixed during the run; group labels never become model features.

Output starts with neutral dark dots at the original input positions. One
purple anchor at a time shows all of its actual NN/MN/FP partners simultaneously.
NN uses RGB(93,205,170), MN RGB(255,171,56), and FP RGB(110,181,232).
Multiple rings and independent colored connections preserve overlapping
memberships and self-membership. Duplicate selected slots remain in counts and
hover text. The model's 10/5/20 pair counts and algorithms are unchanged.

This is paced replay of actual sampler result arrays, one group per anchor.
Candidate-by-candidate checks are no longer animated. Actual Annoy query/distance
records remain in pairs.json for audit; Annoy internal traversal and compiled
MN/FP candidate loops have never been fabricated or exposed by the viewer.

## Estimated whole-demo duration

The numeric seconds field defaults to **90** and accepts any positive target,
without a 60–120 limit. It estimates the entire demo, including preparation,
selection and training replay. Edit it during playback to apply a shorter or longer target immediately. Computation runs independently at normal speed.

After genuine pairs are ready, about 32% of the remaining target is budgeted to
selection replay; remaining time goes to model frames. Actual computation and
demo times are shown separately. Cold preparation, extra detail, a hidden or
delayed browser and very short targets can extend the estimate. A browser cannot
display arbitrarily many distinct states instantly; timing remains approximate.

Every genuine frame, initial through epoch 450, is streamed and played **in
order** at a steady cadence. There is no interpolation, fabricated force model,
burst catch-up or jump straight to the latest epoch. All 451 frames are saved.
The previous observer RNG checks and identical inference-only in-process adapter
remain. Final observed output must equal actual final inference.

The true-input overlay is preserved during training: faint rings at literal
original scores, small solid current outputs, faint center-to-center lines without arrowheads and
matching IDs. Initial-network outputs are not ghost origins. Inputs and raw
embeddings are different spaces on a shared numeric display: lines show
identity correspondence, not meaningful distance or physical motion. Neither
is aligned/transformed. Tight equal-scale bounds and optional collision-aware
labels are retained.

## Reset, files and checks

`Reset / stop` stops this server's owned worker and returns output to neutral
input positions. Input stays visible and prior recordings are retained. Start
then creates a new run. The printed temporary run directory contains live
`state.json`, actual `pairs.json`, append-only `frames.jsonl`, the complete
`epochs.json` archive and `training.log`.

```powershell
node experiment\simulator\test_ui.cjs
& .\.venv\Scripts\python.exe -m unittest discover -s experiment\simulator -p 'test_*.py' -v
```

New evidence: `windows-verification/live-demo/`. Baseline/36, modified/36 and
baseline/100 at seed 42 match their pre-demo desktop final coordinates exactly.
Pair arrays match fresh untouched sampler calls; all 451 delivered HTTP frames
equal the actual archive. Active-worker reset passed in the normal desktop
context after the restricted tool sandbox blocked process-tree termination.

UI tests cover immediate/fixed input, neutral/selection roles, frame order,
cadence, editable duration, transient state-read races, reset/restart, variants,
errors and the true-input overlay. Browser QA is documented separately from
the historical cloud browser limitation. Original cloud verification, reports,
static plots, 60 trials and source algorithms remain unchanged. No paid compute,
public deployment or GitHub changes were used.

## Full-width presentation layout

The fixed input is the first full-width section. Run status (including Demo
complete) follows it; the full-width output is below. Its epoch readout and
progress bar sit directly above the output plot. Small hollow original rings and
solid current/final markers are retained. Faint correspondence lines run fully
from original center to current center, with no arrowheads or endpoint gaps.
Refresh with Ctrl+F5 to load this HTML-only presentation change; no server
restart is required. All pair tracing, duration/cadence controls and computation
are unchanged.

## Grouped anchor selection (latest presentation revision)

Candidate-by-candidate playback has been replaced by one complete anchor view
at a time. Purple marks the source. Every actual NN/MN/FP partner is displayed
simultaneously using NN RGB(93,205,170), MN RGB(255,171,56), and FP
RGB(110,181,232), matching the slide colors. Multiple rings and independent
colored connections retain overlapping memberships, including self-membership.
Repeated selections remain in slot counts and hover text. The event sequence
is one real sampler-result group per anchor, paced by the same editable whole-
demo duration. Earlier candidate distance records remain in pairs.json for audit
but are not animated. All algorithms and training/cadence behavior are unchanged.

## Window-contained graph sizing

Sections remain vertically stacked. Each SVG is proportionally contained by both
its available panel width and a viewport-height budget, with a sensible 900px
maximum plot height. Axes are included in that complete SVG. Height is reserved
for the section heading/caption and epoch controls, so a whole graph fits one
window; navigating between sections can still require vertical scrolling.
Sizing responds automatically when the window changes. Ctrl+F5 loads this
presentation-only update; no server restart is needed.
