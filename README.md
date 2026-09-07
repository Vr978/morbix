# Morbix

Morbix is a symptom-based disease classification system built as the starting phase. Using this repo you will able to completely reproduce it and use a public disease-symptom dataset, clean it, train a classifier, evaluate it, and provide ranked disease predictions from command-line symptom input.

The central research question of this project is: **Can an AI system recognize when the available symptom information is insufficient, determine what additional information would be most useful, and improve disease classification by asking targeted follow-up questions compared with a one-shot classifier?** A standard classifier maps the symptoms it receives directly to a prediction. It does not know whether important information is still missing or what it should ask next. The longer-term goal is to study whether specialized agents can improve this process by interpreting natural-language symptom descriptions, gathering missing information through targeted questions, comparing competing predictions, and verifying the final result before committing to it.

---

## Table of Contents

- [Current Workflow](#current-workflow)
- [Dataset](#dataset)
- [Model](#model)
- [Setup](#setup)
- [Usage](#usage)
  - [Reproducing Everything From the Dataset](#reproducing-everything-from-the-dataset)
  - [Evaluating the Model](#evaluating-the-model)
  - [Generating a Test Case](#generating-a-test-case)
- [Current Gaps and Weaknesses](#current-gaps-and-weaknesses)
- [Planned Direction](#planned-direction)
- [Initial Evaluation Plan](#initial-evaluation-plan)

---

## Current Workflow

```text
User symptoms
     │
     ▼
SymptomAgent
     │
     ▼
DiseasePredictorTool
     │
     ▼
DiseasePredictor
     │
     ▼
Gaussian Naive Bayes model
     │
     ▼
Ranked disease candidates
```

The current agent receives a list of symptoms, cleans the input, and passes it to the prediction tool. The tool invokes the trained classifier and returns the top-ranked disease labels. The current agent performs a single prediction step, it does not ask follow-up questions or revise its answer. Those are the capabilities I am looking for in future versions of the system.

---

## Dataset

The project uses the following public dataset from Hugging Face:

```
dhivyeshrk/Disease-Symptom-Extensive-Clean
```

**Exact revision used:**

```
5f2080be444ba0f43c48b923ffc62d3e3b4897f5
```

### Original Dataset

| Property | Value |
|---|---|
| Rows | 246,945 |
| Symptom columns | 377 |
| Disease labels | 773 |

Symptoms are represented as binary values (`1` = present, `0` = absent). The target column is `diseases`.

### After Cleaning

Before training, the dataset is cleaned by:

- Removing exact duplicate rows
- Removing rows with no active symptoms
- Removing disease classes with fewer than 10 unique samples
- Verifying that all symptom columns contain valid binary values
- Removing symptom columns that are constant (zero variance)

| Property | Value |
|---|---|
| Samples | 188,920 |
| Symptom features | 320 |
| Disease classes | 587 |

### Data Splits

The cleaned data is split using a fixed random seed of `42` with stratification so that every retained disease class is represented across all three sets.

| Split | Samples |
|---|---|
| Training (80%) | 151,136 |
| Validation (10%) | 18,892 |
| Test (10%) | 18,892 |

More information about the data files is available in [`data/README.md`](data/README.md).

---

## Model

The classifier is **Gaussian Naive Bayes** from scikit-learn.

It was chosen because it is simple, lightweight, fast to train locally, and naturally produces scores for multiple disease candidates, which is useful for a system that may need to reason over several possibilities rather than committing immediately to a single answer.

| Property | Value |
|---|---|
| Features used | 320 (all cleaned symptom columns) |
| Classes | 587 disease labels |
| Training samples | 151,136 |

### Validation Results

| Metric | Score |
|---|---|
| Top-1 accuracy | 83.53% |
| Top-3 accuracy | 95.10% |
| Top-5 accuracy | 97.63% |

Top-K metrics are reported because the future system may reason over several candidate diseases rather than using only the top prediction.

---


## Setup

Python 3.10 or newer is recommended.

**1. Clone the repository**

```bash
git clone this repository
cd Morbix
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### Reproducing Everything From the Dataset

**Step 1 — Preprocess:**

```bash
python -m src.preprocess
```

**Step 2 — Train:**

```bash
python -m src.train
```

**Step 3 — Run:**

```bash
python run_solution.py
  --symptoms "cough, fever, fatigue" 
  --show-trace
```

---

### Evaluating the Model

Validation set:

```bash
python3 evaluation/evaluate.py
```

Test set:

```bash
python3 evaluation/evaluate.py --split test
```

---

### Generating a Test Case

Select a real sample from the validation set:

```bash
python3 examples/generate_test_case.py --row 0
```

The script prints the true disease label, active symptoms, and a ready-to-run command. Change `--row` to select a different example:

```bash
python3 examples/generate_test_case.py --row 25
```

---

## Current Gaps and Weaknesses

The most significant weakness is how the system handles incomplete symptom information.

The classifier uses a fixed binary feature vector. Symptoms that are not provided are set to `0`, which is indistinguishable from symptoms that are genuinely absent. The current representation cannot tell the difference between:

- **Symptom is absent** — the patient does not have it
- **Symptom is unknown** — nobody has asked about it yet

This can cause predictions to shift significantly when only a small number of symptoms are provided, because missing information is treated the same way as negative evidence

Additional limitations:

- Symptom names must closely match the dataset vocabulary
- The system has no understanding of alternative descriptions for the same symptom
- A single prediction is made and the process stops
- The system cannot ask follow-up questions or request additional information
- The system cannot identify which unknown symptom would be most useful to ask about next
- The system cannot maintain or compare competing disease candidates
- The system cannot revise its prediction after receiving new information
- There is no independent verification step for the final prediction
- Gaussian Naive Bayes scores are not calibrated probabilities
- Removing classes with too few samples reduces coverage from 773 to 587 disease labels
- Output quality is bounded by the structure and quality of the source dataset

These weaknesses are the primary motivation for extending the system.

---

## Planned Direction

The proposed next phase will investigate whether a multi-agent architecture can improve classification when the initial symptom information is incomplete. The main focus will be whether the system can recognize uncertainty, identify useful missing information, ask targeted follow-up questions, and revise its prediction as new evidence becomes available.

---

## Initial Evaluation Plan

The extended system would be compared against the current baseline using the same dataset and identical starting information.

**Planned metrics:**

| Metric | Description |
|---|---|
| Top-1 accuracy | Fraction of cases where the correct disease is the top prediction |
| Top-3 accuracy | Fraction of cases where the correct disease appears in the top 3 |
| Partial-symptom accuracy | Accuracy when only a subset of symptoms is initially available |
| Post-questions accuracy | Accuracy after a fixed number of follow-up questions |
| Questions to correct result | Average number of questions needed to reach the correct prediction |
| Task completion rate | Fraction of cases where a final prediction is successfully returned |


A representative experiment would start both systems with the same incomplete symptom set. The current baseline would make one prediction immediately, while the agentic system would be allowed a bounded number of follow-up questions before making its final prediction. This makes it possible to measure whether active information gathering actually improves accuracy rather than assuming that a multi-agent architecture is automatically better. If retrieval or other external knowledge sources are later added, additional measures such as faithfulness, context relevance, answer relevance, and cross-source consistency can also be evaluated.
