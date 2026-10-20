# Dataset Registry

> Purpose: determine, **before starting**, whether a research question is experimentally feasible.
> Authoritative sources for this registry: `01_literature/paper-matrix.md`
> (39 papers) and `00_lab/decision-log.md` (C5/C8 protocol freeze, 2026-10-07).
> Any field that is not supported by the literature or the decision log is
> tagged `[TO VERIFY FROM SOURCE DOC]` and must not be used in downstream
> computations before being verified.

## Dataset Card (one per dataset)
```
Dataset:
URL:
Institution / Year:
Machine:
Sensor:
Sampling Rate:
Raw signal: yes/no
Number of machines:
Operating conditions:
Fault types:
Healthy data:
Fault data:
Labels:
Standard train/test protocol:
Publicly available: yes/no
License:
Download size:
Documentation:
Known limitations:
Suitable research questions:
```

## Registry

| # | Name | Machine | Sensor | Available? | License | Size | Source (which paper used it) | Notes |
|---|------|---------|--------|------------|---------|------|------------------------------|-------|
| D1 | CWRU Bearing Data Center | Case Western Reserve University 2-hp induction motor (Drive End + Fan End) | Accelerometer (DE/FE) | Yes | CC-BY 4.0 (public release) | ~120 MB | P15 (Hakam 2026), P16 (El Boughardini 2026), P20 (Cotrino 2026), P38 (Kolok 2025) | Detailed card: `manifests/cwru.md` |
| D2 | Paderborn University Bearing (PU / KAt-Datacenter) | Paderborn Univ. modular test rig (32 bearings) | IEPE vibration + current + temperature | Yes | CC-BY 4.0 (public release) | ~5–10 GB (per the 2024 Zenodo snapshot) | P16 (El Boughardini 2026); 2024 release referenced in `00_lab/decision-log.md` | Detailed card: `manifests/paderborn.md` |
| D3 | Southeast University (SEU) Gearbox | SEU drivetrain dynamometer (gearbox + bearings) | Vibration accelerometer | Yes | CC-BY 4.0 (per MDPI Electronics open-data policy) | `[TO VERIFY FROM SOURCE DOC]` | P27 (Łuczak 2024) | Detailed card: `manifests/seu.md` |
| D4 | (reserved) IMS / NASA / Custom | — | — | — | — | — | — | Reserved for future expansion of the triad. |

> **C5/C8 ruling (frozen 2026-10-07):** the CWRU ↔ PU ↔ SEU triad is the
> official cross-machine benchmark. Any dataset outside the triad requires
> a new entry in `00_lab/decision-log.md` before it is used.

> Note: the Dataset Hunter must capture **every** dataset used in the five
> target papers — not just CWRU. The triad above is complete: 4 + 2 + 1 = 7
> references to benchmark datasets in `01_literature/paper-matrix.md`, which
> matches the corpus statistic "CWRU (4 papers), PU (2 papers), SEU (1 paper)".

---

## Detailed cards (Manifests)

For each dataset in the registry there is a detailed card in `manifests/`:

- `manifests/cwru.md`
- `manifests/paderborn.md`
- `manifests/seu.md`

The cards are written **exclusively** from facts established in
`01_literature/paper-matrix.md` and `00_lab/decision-log.md`. Fields that
cannot be proven from those sources (e.g. exact download size in MB, the
date of the latest Zenodo refresh) are tagged `[TO VERIFY FROM SOURCE DOC]`
and must not be used as inputs to Experiment 002 until verified against the
provider's own page.
