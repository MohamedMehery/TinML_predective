تمام. نبدأ فعليًا، ونعتبر اللي جاي ده **الإصدار الأول من Operating System للمشروع**، مش مجرد قائمة أفكار.

أنا هصممه بحيث إننا نقدر نستخدمه الآن في **TinyML + Predictive Maintenance**، وبعد كده نعيد استخدام نفس المنظومة في أي موضوع Embedded AI آخر.

# Embedded AI Research-to-Impact Lab

## Research Operating System — v0.1

الفكرة الأساسية:

```text
                    ┌─────────────────────┐
                    │   RESEARCH DIRECTOR │
                    │     AI AGENT        │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ↓                    ↓                    ↓
   Literature Agent      Dataset Agent       Gap Hunter
          │                    │                    │
          └──────────────┬─────┴────────────────────┘
                         ↓
                Experiment Designer
                         │
                         ↓
                  Implementation
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
       Embedded/ML Agent       Critical Reviewer
              │                     │
              └──────────┬──────────┘
                         ↓
                 Scientific Writer
                         │
                         ↓
               GitHub / Paper / Article
```

والـ **Research Director** مش لازم يكون Agent مستقل فعليًا من أول يوم. ممكن يكون ChatGPT + مجموعة Prompts منظمة.

---

# 1. أول قاعدة للمشروع

أهم قرار عندي:

> **لن نبدأ بسؤال: "ماذا نستطيع أن نبني؟"**

سنبدأ بـ:

> **"ما المشكلة العلمية المهمة التي ما زالت غير محلولة؟ وهل نستطيع إثبات شيء جديد فيها بالموارد المتاحة لنا؟"**

ثم يأتي المنتج أو الـprototype كنتيجة جانبية للبحث.

وده يحقق اللي أنت قلته:

**Research → Contribution → Prototype → Publication → Network → Opportunity**

وليس:

**Prototype → نحاول نجد له Research Question.**

---

# 2. الـ Repository

أقترح أن يكون الـrepo نفسه هو **Research Lab Notebook**.

مثلاً:

```text
embedded-ai-research-lab/
│
├── README.md
│
├── 00_lab/
│   ├── research-principles.md
│   ├── research-workflow.md
│   ├── decision-log.md
│   └── glossary.md
│
├── 01_literature/
│   ├── papers/
│   ├── paper-matrix.md
│   ├── literature-map.md
│   ├── claims/
│   └── reviews/
│
├── 02_datasets/
│   ├── dataset-registry.md
│   ├── dataset-cards/
│   ├── download-scripts/
│   └── data-quality/
│
├── 03_research_questions/
│   ├── candidate-questions.md
│   ├── gap-analysis.md
│   └── selected-question.md
│
├── 04_experiments/
│   ├── 001_baseline/
│   ├── 002_signal-processing/
│   ├── 003_classical-ml/
│   ├── 004_tinyml/
│   └── 005_embedded-deployment/
│
├── 05_embedded/
│   ├── stm32/
│   ├── zephyr/
│   ├── wokwi/
│   └── benchmarks/
│
├── 06_results/
│   ├── figures/
│   ├── tables/
│   └── experiment-log.md
│
├── 07_publication/
│   ├── reports/
│   ├── linkedin/
│   ├── paper/
│   └── presentation/
│
└── 08_ai_agents/
    ├── director.md
    ├── literature-agent.md
    ├── dataset-agent.md
    ├── gap-hunter.md
    ├── experiment-designer.md
    ├── embedded-agent.md
    ├── reviewer.md
    └── scientific-writer.md
```

لاحظ حاجة مهمة:

**مش هنعمل folders للكود فقط.**

الـrepo هيسجل:

* ماذا قرأنا؟
* ماذا وجدنا؟
* لماذا اخترنا هذا السؤال؟
* لماذا رفضنا سؤالًا آخر؟
* أي dataset استخدمنا؟
* كيف قسمنا البيانات؟
* ماذا كانت النتيجة؟
* ماذا فشل؟
* ماذا تعلمنا؟

وده يخلي الـGitHub repository نفسه جزءًا من الـresearch contribution.

---

# 3. الـAI Agent Team

أنا أقترح **8 Agents**، لكن مش هنشغلهم كلهم مرة واحدة.

## Agent 01 — Research Director

وظيفته:

> يحافظ على اتجاه المشروع بالكامل.

لا يقوم بكل شيء بنفسه.

يقول:

```text
What do we know?
What don't we know?
What evidence do we have?
What should we investigate next?
What task should be delegated?
```

### Prompt أولي

```text
You are the Research Director of an Embedded AI Research Lab.

Your job is NOT to generate random project ideas.

Your job is to guide a research-to-impact workflow:

Literature
→ Evidence
→ Gap
→ Research Question
→ Data Feasibility
→ Experiment
→ Embedded Validation
→ Contribution
→ Publication

Rules:

1. Never assume that a research gap exists without evidence.
2. Separate:
   - what the paper states
   - what the data demonstrates
   - our inference
   - our hypothesis
3. Reject research questions that require inaccessible proprietary data
   unless a realistic alternative exists.
4. Prefer reproducible experiments.
5. Every proposed experiment must define:
   - input data
   - baseline
   - metric
   - expected outcome
   - hardware/software requirements
6. Do not confuse a prototype with a scientific contribution.
7. Challenge confirmation bias.
8. Identify claims that require independent verification.

For every research task, produce:

- Objective
- Known evidence
- Unknowns
- Hypotheses
- Required evidence
- Next actions
- Risks
- Decision
```

---

# 4. Literature Agent

ده اللي هيمسك الأوراق الخمسة.

لكن مش عايزه يعمل:

> "Paper summary"

فقط.

عايزه يعمل **Evidence Extraction**.

لكل ورقة:

```text
Research Question
Dataset
Sensors
Machine
Operating Conditions
Preprocessing
Features
Model
Training
Validation
Deployment
MCU
RAM
Flash
Latency
Energy
Accuracy
Generalization
Limitations
Future Work
```

ثم يسأل:

> **What does this paper actually prove?**

وده سؤال خطير ومهم جدًا.

---

# 5. Dataset Hunter

وده عندي من أهم الـAgents في المشروع.

وظيفته ليست فقط:

> Find datasets.

بل:

> **Determine whether a research question is experimentally feasible.**

لكل Dataset يعمل بطاقة:

```text
Dataset:
URL:
Institution:
Year:
Machine:
Sensor:
Sampling Rate:
Raw Signal:
Number of Machines:
Operating Conditions:
Fault Types:
Healthy Data:
Fault Data:
Labels:
Train/Test Protocol:
Publicly Accessible:
License:
Download Size:
Documentation:
Known Limitations:
Suitable Research Questions:
```

وبالتالي قبل ما نقع في المشكلة اللي أنت خايف منها:

> "أنا محتاج vibration data لموتور معين ومش لاقيها."

الـDataset Agent يقول لنا **قبل البداية** هل السؤال قابل للتنفيذ أم لا.

---

# 6. Gap Hunter

ده هيكون من أخطر الوكلاء.

مش مهمته يقول:

> "هناك gap في research."

دي جملة سهلة جدًا.

مهمته:

### يعمل Cross-Paper Analysis

مثلاً:

| Problem                        | Paper 1 | Paper 2 | Paper 3 | Paper 4 | Paper 5 |
| ------------------------------ | ------: | ------: | ------: | ------: | ------: |
| Tiny MCU deployment            |       ✓ |       ✓ |       ✓ |       ✓ |       ✓ |
| Quantization                   |       ✓ |       ✓ |       ✓ |       ✓ |         |
| Different operating conditions |         |         |       ? |         |       ? |
| Cross-machine validation       |         |         |         |         |         |
| Unknown faults                 |         |         |         |         |         |
| Online adaptation              |         |         |         |         |         |
| Energy measurement             |         |         |         |         |         |
| Standardized benchmark         |         |         |         |         |         |
| Real industrial deployment     |         |         |         |         |         |

ثم يبدأ يسأل:

> What is repeatedly claimed but insufficiently demonstrated?

**وهنا بالضبط يبدأ البحث الحقيقي.**

---

# 7. Experiment Designer

بعد اختيار Research Question.

وظيفته تحويل السؤال إلى تجربة.

مثلاً:

```text
Research Question:
Does quantization preserve fault-detection
performance under constrained MCU resources?
```

يحولها إلى:

```text
Dataset
↓
Train baseline
↓
FP32 model
↓
INT8 model
↓
Compare accuracy
↓
Measure model size
↓
Measure RAM
↓
Measure inference latency
↓
Deploy
↓
Measure real execution
```

والأهم:

**يحدد الـbaseline قبل النموذج الجديد.**

لأن بدون baseline ممكن نعمل نتيجة تبدو ممتازة وهي في الحقيقة لا تعني شيئًا.

---

# 8. Embedded Agent

وده المكان اللي نستخدم فيه خبرتك.

هو مسؤول عن:

* STM32
* Zephyr
* CMSIS
* DSP
* TensorFlow Lite Micro / alternatives
* memory
* latency
* DMA
* buffering
* sampling
* real-time constraints
* Wokwi

لكن عنده قاعدة:

> **لا يقرر السؤال العلمي.**

هو ينفذ ويقيس.

---

# 9. Critical Reviewer

أنا أريد ده يكون Agent إجباري.

بعد كل نتيجة، لا يسمح لنا بالاحتفال مباشرة 😂

يسأل:

```text
Could this result be caused by data leakage?

Could train/test samples come from the same machine?

Could window overlap inflate accuracy?

Does the test represent a new operating condition?

Is accuracy the correct metric?

Was the baseline strong enough?

Are we measuring real-time performance or only inference time?

Does the experiment support the paper's conclusion?

What alternative explanation exists?
```

وده مهم جدًا لأن جزءًا من القيمة التي يمكن أن تضيفها أنت ليس اختراع CNN جديد.

ممكن تكون القيمة:

> **كشف أن طريقة تقييم شائعة تعطي نتائج مضللة تحت ظروف معينة.**

وده بحث محترم جدًا لو أثبتناه جيدًا.

---

# 10. Scientific Writer

في الآخر يحول:

```text
Evidence
+
Experiments
+
Results
+
Limitations
```

إلى:

* Technical report
* Research article
* LinkedIn article
* GitHub documentation
* presentation
* proposal للجامعة/باحث

لكن **لا يكتب قبل أن يكون عندنا evidence**.

---

# 11. الآن عندنا Parallel Tasks

وده بالضبط اللي كنت بتقوله من شوية.

أنا متفق معاك إننا **لا نمشي خطيًا**.

نعمل Parallel Workstreams:

### Track A — Research

```text
Paper 1
Paper 2
Paper 3
Paper 4
Paper 5
↓
Literature Matrix
↓
Gap Analysis
↓
Research Questions
```

### Track B — Data

```text
Dataset Discovery
↓
Dataset Registry
↓
Download
↓
Quality Assessment
↓
Reproducibility
```

### Track C — Environment

```text
Python
Jupyter
ML libraries
DSP
TinyML framework
Zephyr
STM32
Wokwi
Git
```

### Track D — Knowledge Infrastructure

```text
GitHub
Documentation
Experiment templates
AI Agent prompts
Decision log
Research notes
```

### Track E — Communication

**لسه مش هننشر نتائج.**

لكن نبدأ نجمع:

```text
Interesting finding
Interesting contradiction
Interesting question
Unexpected result
Research insight
```

عشان بعدين تتحول إلى content.

---

# 12. والـSetup مش انتظار

ودي النقطة اللي أنت كنت بتقولها.

**نعم، الـsetup يكون Parallel Task.**

لكن مش نجهز كل حاجة في العالم.

نعمل:

### Setup 0 — Research Environment

```text
Git
GitHub
Python
Jupyter
Pandas
NumPy
SciPy
scikit-learn
Matplotlib
```

### Setup 1 — Embedded

```text
VS Code
Zephyr
west
ARM GCC
Wokwi
STM32F103
```

### Setup 2 — TinyML

نحدد framework بعد ما نعرف احتياجات التجربة، بدل ما نختار framework لمجرد أنه مشهور.

---

# 13. أول Sprint

أنا لو أنا اللي هقود المشروع، **مش هبدأ coding بكرة.**

أول Sprint عندي سيكون:

## Sprint 01 — From Papers to Research Map

### Task 01

تحليل الأوراق الخمسة.

### Task 02

إنشاء:

`paper-matrix.md`

### Task 03

إنشاء:

`literature-map.md`

### Task 04

إنشاء:

`gap-analysis.md`

### Task 05 — بالتوازي

Dataset Hunter يبحث عن **كل datasets المستخدمة في الأوراق الخمسة**.

وليس مجرد CWRU ونخلص.

### Task 06

إنشاء:

`dataset-registry.md`

### Task 07

تحديد:

> Which research questions can we realistically test with public data?

---

# 14. والـDeliverable الأول

مش Model.

مش ESP32.

مش CNN.

مش PCB.

الـDeliverable الأول سيكون شيئًا اسمه:

# Research Intelligence Brief

مثلاً:

> **TinyML for Machinery Fault Diagnosis:
> From Academic Prototypes to Reliable Edge Deployment**

ويكون مبنيًا على الأوراق الخمسة.

وفيه:

1. ماذا يحدث في المجال؟
2. ماذا أثبتت الأبحاث؟
3. ماذا لم تثبته؟
4. أين تتكرر المشاكل؟
5. ما الـdatasets المتاحة؟
6. ما الذي يمكن إعادة إنتاجه؟
7. ما الفجوات المحتملة؟
8. ما الأسئلة التي يمكن لباحث مستقل تنفيذها؟
9. ما السؤال الأكثر promising؟
10. ما التجربة التي يمكن أن تثبته؟

---

# 15. وبعدها نختار Contribution

وهنا أريدك تركز في الفرق:

**Contribution ≠ Product**

ممكن الـcontribution يكون:

* طريقة تقييم أفضل.
* Benchmark جديد.
* مقارنة منهجية.
* إثبات limitation.
* تحسين robustness.
* تقليل memory مع الحفاظ على الأداء.
* طريقة preprocessing أفضل.
* adaptive inference.
* deployment methodology.
* reproducible framework.
* dataset transformation.
* تحليل trade-off بين accuracy/latency/energy.

والـprototype يأتي لإثبات الـcontribution.

---

## إذن قرارنا من الآن

أنا أقترح أن يكون **أول مشروع فعلي لنا داخل الـLab هو نفسه بناء الـLab**.

ثم أول Research Sprint:

```text
             5 PAPERS
                 │
                 ▼
        ┌────────────────┐
        │ PAPER MATRIX   │
        └───────┬────────┘
                ▼
        ┌────────────────┐
        │ LITERATURE MAP │
        └───────┬────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
 DATASET AUDIT       GAP ANALYSIS
       │                 │
       └────────┬────────┘
                ▼
       RESEARCH QUESTIONS
                │
                ▼
       FEASIBILITY FILTER
                │
                ▼
       SELECT CONTRIBUTION
                │
                ▼
           EXPERIMENT
                │
                ▼
       EMBEDDED VALIDATION
                │
                ▼
     REPORT / PAPER / GITHUB
                │
                ▼
       UNIVERSITY / RESEARCHER
       / GRANT / JOB NETWORK
```

**وده هو الـKickoff الحقيقي في رأيي.**

والخطوة التنفيذية التالية التي سأقودك فيها هي **مش كتابة كود**: هننشئ أولًا ملفات الـrepo الأساسية + الـtemplates الخاصة بالـPaper Matrix والـDataset Registry والـGap Analysis + Prompts للـAgents، بحيث من أول يوم يبقى عندنا نظام شغال، مش مجرد محادثة ChatGPT.
