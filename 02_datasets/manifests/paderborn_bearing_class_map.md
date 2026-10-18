# Paderborn bearing class map

## Purpose and scope

This manifest records the owner-approved Stage 1 class encoding for the 32 Paderborn University bearing IDs listed in Lessmeier et al. (2016), together with the source table and the source component/combination/extent fields needed to trace each row. The CSV is the row-level source of truth; this document records its label contract and encoding conventions.

This is documentation and test coverage only. It does not change a loader, runtime behavior, training procedure, or experiment. Class names and labels below are the owner-provided encoding; they are not inferred from filename prefixes. The paper is used to verify the bearing IDs, table references, and recorded source fields, not to silently replace the owner's requested Stage 1 policy.

## Source and verification record

**Citation:** Christian Lessmeier, James Kuria Kimotho, Detmar Zimmer, and Walter Sextro (2016), “Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification,” *European Conference of the Prognostics and Health Management Society 2016* (PHME 2016), 3(1), DOI [10.36001/phme.2016.v3i1.1577](https://doi.org/10.36001/phme.2016.v3i1.1577).

**Repository source artifact:** `01_literature/papers/Lessmeier 2016.pdf`, read from commit `359ac7f068c65b87a396f86da22abe50a82140ad` (`origin/master`); blob `2951a6a5d8d10cca8f9d974709c851110310d89f`. The PDF was inspected from the already-present Git object; no paper or dataset was downloaded.

The source cross-check found no conflict with the owner-provided bearing/component/extent mapping:

- **Table 4 (PDF p. 5):** the 12 artificial-damage bearing IDs, components, and extent values match the CSV. The sentence following the table states that all artificial damages are single-point damages without repetition or combination.
- **Table 5 (PDF p. 6):** the 14 accelerated-lifetime real-damage bearing IDs and the recorded bearing-element, combination-code, and extent values match the CSV. The printed legend defines `S` as single damage, `R` as repetitive damage, and `M` as multiple damage.
- **Table 7 (PDF p. 8):** the six healthy/undamaged bearing IDs are K001–K006, matching the CSV.
- **Section 5 (PDF p. 9):** the paper states that the dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0), says noncommercial academic use is explicitly allowed with citation, and directs commercial users to contact the author. This is the paper's statement about the **data** license. The article's separate open-access copyright notice identifies CC BY 3.0 US for the article itself; the two statements are not conflated here.

## Label contract and Stage 1 status

| `label` | `class_name` | Meaning in this manifest | `stage1_status` |
|---:|---|---|---|
| `0` | `Healthy` | Healthy/undamaged reference bearing | `include` |
| `1` | `Outer_Race` | Owner-mapped outer-race class | `include` |
| `2` | `Inner_Race` | Owner-mapped inner-race class | `include` |
| `-1` | `Combined` | Documentation category for the three owner-designated combined-damage bearings; **not** a fourth Stage 1 class | `exclude` |

`stage1_status=include` means the row has one of the three Stage 1 labels 0, 1, or 2. `stage1_status=exclude` means the row is deliberately excluded from Stage 1 and carries label `-1`. The owner-designated combined-damage bearings KB23, KB24, and KB27 are therefore explicitly excluded, not collapsed into either race class.

## CSV field conventions

The CSV uses the exact requested columns, in this order:

`bearing_id,class_name,label,damage_origin,source_table,source_component,source_combination,extent_level,stage1_status`

- `bearing_id` is copied from the cited table, not derived from a filename prefix.
- `damage_origin` is `healthy`, `artificial`, or `real`, following the source grouping and the owner mapping.
- `source_table` identifies Table 4, 5, or 7 for each row.
- For healthy rows, `source_component`, `source_combination`, and `extent_level` are `n/a`, as specified.
- For Table 4, `source_component` preserves the table's `OR`/`IR` entry. Table 4 has no combination column; consistently, `source_combination` is recorded as the exact phrase `single point`, based on the explicit sentence after Table 4 describing all its artificial damages. This is a documentation convention for that source statement, not an additional physical interpretation.
- For Table 5, `source_component` preserves the source bearing-element text, including the distinctions `IR (+OR)` and `OR + IR`; `source_combination` preserves the source's literal `S`, `R`, or `M` code. The source's arrangement column is not part of the requested CSV schema. `M` remains a source combination code: it is not converted into a class and, by itself, is not used to infer a multi-component interpretation.
- `extent_level` preserves the owner-provided strings `L1`, `L2`, and `L3`, corresponding to the numeric extent entries in Tables 4 and 5. No severity meaning is added to `S`, `R`, or `M`.

## Row counts

| Source grouping | Rows | Included in Stage 1 | Excluded from Stage 1 |
|---|---:|---:|---:|
| Healthy, Table 7 | 6 | 6 | 0 |
| Artificial damage, Table 4 | 12 | 12 | 0 |
| Real damage, Table 5 | 14 | 11 | 3 |
| **Total** | **32** | **29** | **3** |

The three excluded rows are KB23, KB24, and KB27. `Combined` is documentation-only and is not a Stage 1 training class.
