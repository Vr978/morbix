# Morbix

Morbix is a symptom-based disease classification system built as the starting phase of this project. Using this repository, the complete pipeline can be reproduced from a public disease-symptom dataset, including data preprocessing, model training, evaluation, and ranked disease prediction from command-line symptom input.

The central research question is:

**Can an AI system recognize when the available symptom information is insufficient, determine what additional information would be most useful, and improve disease classification by asking targeted follow-up questions compared with a one-shot classifier?**

A standard classifier maps the symptoms it receives directly to a prediction. It does not know whether important information is still missing or what should be asked next. The longer-term goal is to study whether an agentic system can support doctors or trained medical staff by interpreting symptom descriptions, gathering missing information through targeted questions, comparing possible disease candidates, and updating its prediction as new evidence becomes available.

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
     |
     v
SymptomAgent
     |
     v
DiseasePredictorTool
     |
     v
DiseasePredictor
     |
     v
Gaussian Naive Bayes model
     |
     v
Ranked disease candidates
```

The current agent receives a list of symptoms, cleans the input, and passes it to the prediction tool. The tool converts the recognized symptoms into the feature representation expected by the trained classifier and returns the highest-ranked disease labels.

The current system performs one prediction step. It does not yet ask follow-up questions, track unknown information, or revise its prediction after receiving additional evidence.

---

## Dataset

The project uses the following public Hugging Face dataset:

```text
dhivyeshrk/Disease-Symptom-Extensive-Clean
```

**Exact revision used:**

```text
5f2080be444ba0f43c48b923ffc62d3e3b4897f5
```

### Original Dataset

| Property | Value |
|---|---:|
| Rows | 246,945 |
| Symptom columns | 377 |
| Disease labels | 773 |

Symptoms are represented using binary values:

```text
1 = symptom present
0 = symptom absent
```

The target column is `diseases`.

### After Cleaning

Before training, the dataset is processed by:

- removing exact duplicate rows
- removing rows with no active symptoms
- removing disease classes with fewer than 10 unique samples
- verifying that symptom columns contain valid binary values
- removing constant symptom columns

| Property | Value |
|---|---:|
| Samples | 188,920 |
| Symptom features | 320 |
| Disease classes | 587 |

### Data Splits

The cleaned dataset is split using a fixed random seed of `42` with stratification so that every retained disease class is represented in all three sets.

| Split | Samples |
|---|---:|
| Training, 80% | 151,136 |
| Validation, 10% | 18,892 |
| Test, 10% | 18,892 |

More information about the generated data files is available in [`data/README.md`](data/README.md).

---

## Model

The current classifier uses **Gaussian Naive Bayes** from scikit-learn.

It was chosen because it is simple, lightweight, fast to train locally, and provides class scores that can be used to rank multiple disease candidates.

| Property | Value |
|---|---:|
| Features used | 320 |
| Disease classes | 587 |
| Training samples | 151,136 |

### Validation Results

| Metric | Score |
|---|---:|
| Top-1 accuracy | 83.53% |
| Top-3 accuracy | 95.10% |
| Top-5 accuracy | 97.63% |

Top-K metrics are useful because an extended system may reason over several possible diseases instead of immediately committing to only the highest-ranked prediction.

---

## Setup

Python 3.10 or newer is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/Vr978/morbix.git
cd Morbix
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

No API keys or environment variables are required for the current system.

---

## Usage

If `models/disease_model.joblib` is already available, the system can be run directly:

```bash
python3 run_solution.py \
  --symptoms "cough, fever, fatigue" \
  --show-trace
```

A different number of ranked candidates can be requested using:

```bash
python3 run_solution.py \
  --symptoms "cough, fever, fatigue" \
  --top-k 3
```

### Reproducing Everything From the Dataset

The full pipeline can also be reproduced from the original dataset.

#### Step 1: Preprocess

```bash
python3 -m src.preprocess
```

This downloads the dataset, cleans it, creates the train, validation, and test splits, and stores the generated files in `data/processed/`.

#### Step 2: Train

```bash
python3 -m src.train
```

This trains the Gaussian Naive Bayes classifier and creates:

```text
models/disease_model.joblib
models/validation_metrics.json
```

#### Step 3: Run

```bash
python3 run_solution.py \
  --symptoms "cough, fever, fatigue" \
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

The evaluation reports Top-1, Top-3, and Top-5 accuracy.

---

### Generating a Test Case

A real validation example can be generated using:

```bash
python3 examples/generate_test_case.py --row 0
```

The script prints the expected disease label, active symptoms, and a ready-to-run command.

A different example can be selected by changing the row number:

```bash
python3 examples/generate_test_case.py --row 25
```

---

## Current Gaps and Weaknesses

The main limitation of the current system is how it handles incomplete symptom information.

The classifier uses a fixed binary feature vector. Symptoms that are provided are set to `1`, while symptoms that are not provided remain `0`. This means the current representation cannot distinguish between:

- **Symptom is absent:** the patient was asked and does not have the symptom.
- **Symptom is unknown:** the symptom has not been asked about yet.

As a result, missing information can be treated the same way as negative evidence. This can affect predictions when only a small number of symptoms are initially available or when multiple diseases share similar symptom patterns.

Additional limitations include:

- symptom names need to closely match the dataset vocabulary
- alternative or natural-language descriptions of the same symptom may not be recognized
- the system makes one prediction and stops
- it cannot determine whether more information is needed
- it cannot choose which unknown symptom would be most useful to ask about
- it does not track present, absent, and unknown symptoms separately
- it cannot revise its prediction after receiving new information
- there is no independent verification step before returning the ranked candidates
- Gaussian Naive Bayes scores should not be treated as calibrated confidence values
- removing classes with too few unique samples reduces coverage from 773 to 587 disease labels
- prediction quality is limited by the structure and quality of the source dataset

These limitations provide the main motivation for extending the current system.

---

## Planned Direction

The next phase will investigate whether an agentic or multi-agent system can improve assistive disease classification when the initial symptom information is incomplete.

The main focus will be whether the system can:

- interpret natural-language symptom descriptions
- recognize when the available evidence is insufficient
- track present, absent, and unknown symptoms
- identify useful missing information
- ask targeted follow-up questions
- update disease candidates as new evidence becomes available
- verify the final ranked prediction before presenting it to medical staff

The goal is not simply to add multiple agents, but to determine whether active information gathering and multiple reasoning steps provide measurable improvement over the current one-shot approach.

---

## Initial Evaluation Plan

The extended system will be compared with the current one-shot system using the same dataset and identical starting symptom information.

| Metric | Description |
|---|---|
| Top-1 accuracy | Correct disease is the highest-ranked prediction |
| Top-3 accuracy | Correct disease appears among the top three predictions |
| Partial-symptom accuracy | Accuracy when only part of the symptom information is initially available |
| Post-question accuracy | Accuracy after a fixed number of follow-up questions |
| Questions to correct result | Number of questions required before reaching the correct prediction |
| Task completion rate | Cases where the system successfully produces a final result |
| Latency | Time required to reach the final prediction |

A representative experiment would start both systems with the same incomplete symptom set. The current one-shot system would make a prediction immediately, while the extended system would be allowed a limited number of follow-up questions before returning its final ranked candidates.

Improvement would be demonstrated if the extended system achieves better classification accuracy using a small number of relevant follow-up questions without introducing excessive latency or unnecessary interactions.

If external retrieval or additional knowledge sources are later introduced, evaluation can be expanded to include measures such as faithfulness, context relevance, answer relevance, and cross-source consistency.