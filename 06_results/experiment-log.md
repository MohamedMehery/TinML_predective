# Experiment Log

| Experiment | Question | Data | Baseline | Model | Result | Failed? | What we learned |
|------------|----------|------|----------|-------|--------|---------|-----------------|
| 001-thermal-audit | Do transformer oil-temperature trip alarms (OTI_T) represent a physically predictable thermal degradation? | Overview.csv + CurrentVoltage.csv (19.3k rows) | IEC 60076-7 Top-Oil ODE (Validation RMSE 1.44°C) | Physics Residual Innovation Filter | The degradation assumption failed: OTI_T is just a threshold >= 236°C (a complete gap across 166°C with 0 samples), and the cooling from 248 to 52 in 8 minutes requires 233 kW — physically impossible in ONAN. | Yes (decisive Negative Result) | Predicting OTI_T with AI is a fallacy because the event is an electronics fault in the sensor circuit. The correct TinyML solution is a rate-of-change sensor-fault detector (calibrated at p99.9 = 2.0°C/min). |

> Rule: failure is logged with the same weight as success. A well-documented Negative Result is respectable research.
