# Experiments

Each experiment = a numbered folder of the form:
```
0NN_<name>/
├── README.md        (Question, hypothesis, data, Baseline, metric)
├── config/          (all settings — no hard-coded values in code)
├── src/
├── results/
└── NOTES.md         (what failed, what we learned)
```

## Rules
1. No experiment without a pre-defined Baseline.
2. No celebrating a result before it has passed the Critical Reviewer.
3. Every experiment must be re-runnable with a single command.
