# Data

The project uses the following public disease-symptom dataset from Hugging Face:

```
dhivyeshrk/Disease-Symptom-Extensive-Clean
```

**Exact revision used:**

```
5f2080be444ba0f43c48b923ffc62d3e3b4897f5
```

The dataset is downloaded automatically by the preprocessing script. Neither the original nor the processed data needs to be stored in the repository.

---

## Original Dataset

| Property | Value |
|---|---|
| Rows | 246,945 |
| Symptom columns | 377 |
| Disease labels | 773 |

The target column is `diseases`. All remaining columns represent symptoms as binary values:

- `1` — symptom is present
- `0` — symptom is absent

---

## Preprocessing

Run the preprocessing script to download and clean the dataset:

```bash
python -m src.preprocess
```

The script performs the following steps:

1. Downloads the dataset from Hugging Face at the pinned revision
2. Cleans disease label formatting
3. Validates that all symptom columns contain only `0` and `1`
4. Removes exact duplicate rows
5. Removes rows with no active symptoms
6. Removes disease classes with fewer than 10 unique samples
7. Removes symptom columns that contain only one value (zero variance)
8. Creates stratified training, validation, and test sets

### After Cleaning

| Property | Value |
|---|---|
| Samples | 188,920 |
| Symptom features | 320 |
| Disease classes | 587 |

### Data Splits

Splits use a fixed random seed of `42` with stratification so that every retained disease class appears in all three sets.

| Split | Samples |
|---|---|
| Train (80%) | 151,136 |
| Validation (10%) | 18,892 |
| Test (10%) | 18,892 |

---