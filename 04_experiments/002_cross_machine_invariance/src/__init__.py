"""
Experiment 002 (EXP-C5C8-002) — pipeline package.

Frozen config: ../config/experiment_config.json
Charter:       ../../../03_research_questions/candidate-questions.md
Decision log:  ../../../00_lab/decision-log.md
NOTES:         ../NOTES.md

This package implements the 5-representation pipeline (R1_Raw, R1_Env,
R2_Raw, R2_Env, R3_Order) with:
  - 3-class contract (Healthy / Outer_Race / Inner_Race); Ball is filtered
    out before windowing (see datasets.py).
  - Fixed-grid angular resampling for R3_Order (256 samples for CWRU's
    12 kHz case, 64 for PU's 64 kHz case; never mixed within a dataset).
  - Strict Leave-One-Bearing-Out (LOBO) splitting on physical_bearing_uid.
  - Source-train-only z-score normalization.
  - Multi-seed training (5 seeds per the config) with cluster bootstrap
    on physical_bearing_uid (n=10,000) for the summary stats.
"""
