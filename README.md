# Embedded AI Research-to-Impact Lab

Research Lab Notebook — TinyML + Predictive Maintenance (reusable for any Embedded AI topic).

## Core Principle
We do not start with "What can we build?" — we start with "What unsolved scientific problem can we prove is solvable with the resources we have?"

Path: Research → Contribution → Prototype → Publication → Network → Opportunity

## Repository Structure
- `00_lab/` — Principles, workflow, decision log
- `01_literature/` — Papers, comparison matrix, literature map, critical analysis
- `02_datasets/` — Dataset registry and per-dataset cards
- `03_research_questions/` — Candidate and selected research questions
- `04_experiments/` — Numbered experiments (001, 002, ...)
- `05_embedded/` — STM32 / Zephyr / Wokwi / benchmarks
- `06_results/` — Numbers, figures, experiment log
- `07_publication/` — Reports, articles, talks
- `08_ai_agents/` — Agent prompts

## Commit Rule
Every file that changes = one clear commit. Record decisions in `00_lab/decision-log.md` before any large step.

## Paderborn (KAt) MAT Reader — Protocol-Neutral Loader

`tools/paderborn_mat_reader/recording.py` provides `load_pu_recording(path, *, speed_unit)`,
which reuses the isolated reader and returns a validated, protocol-neutral
data-loading interface. `speed_unit="rpm_assumed"` must be passed explicitly.

Returned fields: `vibration` (float64), `vibration_time` (retained original
timestamps), `speed_hz` (RPM converted to Hz and linearly interpolated at
retained vibration timestamps), `fs_nominal_hz` (64000), `fs_endpoint_hz`,
and `metadata` (bearing_id, condition code, nominal RPM mapping N15→1500 /
N09→900, original-axis max abs residual, dropped counts, assumptions).

Validation gates (on original axes): endpoint rate within 0.1% of 64000 Hz,
max abs timestamp residual < 1/64000 s, finite and strictly increasing
timestamps with matching lengths, and speed finite with mean within 5% of
nominal. Common time support truncation is applied; no extrapolation.
