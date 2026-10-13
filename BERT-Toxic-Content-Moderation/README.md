# BERT Toxic Content Moderation

A BERT-based machine learning project for detecting toxic and harmful comments through multi-label text classification. This project includes text preprocessing, model training, evaluation artifacts, per-label threshold tuning, and inference using a saved model.

**Author:** Chaitanya
**GitHub:** [@chaitu4206](https://github.com/chaitu4206)

## Overview

Online platforms need reliable ways to identify harmful comments while reducing false positives. This project uses a fine-tuned BERT model to classify text across six toxicity categories:

- **Toxic** — generally abusive or harmful language
- **Severe Toxic** — highly abusive language
- **Obscene** — obscene or vulgar language
- **Threat** — threatening language
- **Insult** — insulting or demeaning language
- **Identity Hate** — hateful language targeting an identity group

This is a **multi-label classification** task, meaning a single comment can belong to multiple categories simultaneously.

## Features

- Text cleaning and preprocessing
- BERT-based multi-label text classification
- Training with weighted binary cross-entropy loss
- Multilabel-stratified dataset splitting
- Per-label decision threshold tuning
- Saved evaluation predictions and thresholds
- Local inference using the included model
- Precision, recall, and F1-score evaluation

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| NLP and transformer models | Hugging Face Transformers |
| Dataset utilities | Hugging Face Datasets |
| Data processing | Pandas, NumPy |
| Evaluation | Scikit-learn |
| Model architecture | BERT (`bert-base-uncased`) |
| Version control | Git and Git LFS |

## Repository Structure

```text
BERT-Toxic-Content-Moderation/
├── data/
│   └── train.csv
├── models/
│   ├── final/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── evaluation-v3/
│       ├── best_thresholds.npy
│       └── results.npz
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── threshold_tuning.py
│   └── train.py
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

The dataset is not included in the repository. The final inference model and selected evaluation artifacts are tracked in Git, with model weights stored through Git LFS. Other local checkpoints and training outputs may not be included.

## Dataset

This project uses the **Jigsaw Toxic Comment Classification Challenge** dataset.

Dataset source: [Jigsaw Toxic Comment Classification Challenge — Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

### Setup

1. Open the Kaggle dataset page.
2. Sign in and download the competition data.
3. Extract the archive.
4. Place `train.csv` inside the project's `data/` directory.

Expected path:

```text
data/train.csv
```

Required columns:

```text
comment_text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

The preprocessing code also supports alternative text-column names such as `text` and `comment`, but all six label columns are required.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/chaitu4206/BERT-Toxic-Content-Moderation.git
cd BERT-Toxic-Content-Moderation
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run Python through the virtual environment directly or use Command Prompt.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 4. Download the dataset

Follow the dataset instructions above and ensure `data/train.csv` exists before running training.

## Text Preprocessing

The preprocessing module performs the following operations:

1. Converts text to lowercase.
2. Removes URLs.
3. Removes HTML tags.
4. Removes `@username` mentions.
5. Normalizes repeated whitespace.
6. Removes rows with missing text.
7. Removes duplicate comments based on the original text column.
8. Validates required label columns.
9. Removes comments that become empty after cleaning.

Implementation: `src/preprocessing.py`

## Model Training

The training script uses the pretrained `bert-base-uncased` model for six-label classification.

| Parameter | Value |
|---|---:|
| Pretrained model | `bert-base-uncased` |
| Maximum sequence length | 64 tokens |
| Dataset sample | 20,000 comments |
| Training set | 16,000 comments |
| Validation set | 2,000 comments |
| Test set | 2,000 comments |
| Training epochs | 1 |
| Training batch size | 4 |
| Gradient accumulation steps | 4 |
| Learning rate | 0.00002 |
| Weight decay | 0.01 |
| Random seed | 42 |

The script samples 20,000 records and uses multilabel-stratified splitting to create the training, validation, and test sets. Weighted binary cross-entropy loss is used to help address label imbalance, and decision thresholds are tuned using validation data.

### Run training

From the repository root:

```powershell
cd src
python train.py
```

The script is configured to write its V3 training model to:

```text
models/bert-toxic-model-v3/
```

Evaluation outputs and tuned thresholds are written to:

```text
models/evaluation-v3/
```

Training requires the dataset and may take considerable time depending on the available hardware. Mixed-precision training is enabled when CUDA is available.

**Important:** Retraining writes to the V3 output directory. It does not automatically replace the model in `models/final/`, which is used by the included inference script.

## Inference

The inference script loads the model from `models/final/` and uses the saved per-label thresholds in `models/evaluation-v3/best_thresholds.npy`.

Run it from the `src` directory:

```powershell
python inference.py
```

The model is configured to load local files, so the required model and tokenizer files must be present in `models/final/`.

If inference fails, check that:

- Git LFS model weights have been downloaded correctly.
- Dependencies in `requirements.txt` are installed.
- Model and tokenizer files exist in `models/final/`.
- The saved threshold file exists in `models/evaluation-v3/`.

The exact interaction and output format depend on the implementation in `src/inference.py`.

## Evaluation Results

The following results were recalculated from the saved test labels and predictions in `models/evaluation-v3/results.npz`.

### Evaluation Summary

| Metric | Score |
|---|---:|
| Test samples | 2,000 |
| Micro F1-score | 0.6922 |
| Macro F1-score | 0.5029 |

### Per-Label Performance

| Label | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Toxic | 0.7892 | 0.7684 | 0.7787 |
| Severe Toxic | 0.4138 | 0.6000 | 0.4898 |
| Obscene | 0.6690 | 0.8879 | 0.7631 |
| Threat | 0.0000 | 0.0000 | 0.0000 |
| Insult | 0.5762 | 0.8614 | 0.6905 |
| Identity Hate | 0.1884 | 0.6842 | 0.2955 |

These metrics describe the saved predictions and labels, not a newly retrained model. The threat category's zero F1-score indicates that the saved predictions did not correctly identify any positive threat examples under the thresholds used.

The difference between micro and macro F1 highlights the challenge of imbalanced multi-label classification. Performance varies significantly between categories, so improvements should focus on minority labels and false-positive/false-negative trade-offs.

## Saved Evaluation Artifacts

- `models/evaluation-v3/results.npz` — saved probabilities, binary predictions, and ground-truth labels for the 2,000 test samples.
- `models/evaluation-v3/best_thresholds.npy` — saved per-label decision thresholds.

| Label | Threshold |
|---|---:|
| Toxic | 0.75 |
| Severe Toxic | 0.80 |
| Obscene | 0.70 |
| Threat | 0.55 |
| Insult | 0.65 |
| Identity Hate | 0.40 |

Changing thresholds can change precision, recall, and F1-score without retraining the model.

## Reproducibility Notes

For more consistent experiments:

1. Use the dependency versions specified in `requirements.txt`.
2. Download the expected dataset and place it at `data/train.csv`.
3. Keep the configured random seed unchanged.
4. Use the same data-sampling and splitting procedure.
5. Record changes to the model, thresholds, or training configuration.
6. Distinguish the included inference model from models produced by subsequent training runs.

Exact numerical reproduction may still vary with hardware, library behavior, and training environment. The included evaluation artifacts preserve one set of predictions for inspection.

## Limitations

- Toxicity labels can be subjective and context-dependent.
- Minority labels may have substantially lower performance.
- A classification score should not be treated as a definitive judgment about a person or their intent.
- Thresholds may need adjustment for different platforms and moderation policies.
- The model should be evaluated on representative data before real-world use.
- Automated moderation decisions should include appropriate review and appeal mechanisms where practical.

## Future Improvements

- Improve recall and precision for underrepresented toxicity categories.
- Compare alternative threshold-tuning strategies.
- Evaluate performance on unseen and out-of-domain comments.
- Add confusion matrices and per-label precision-recall curves.
- Compare model versions using a consistent held-out test set.
- Explore human review for uncertain or high-impact moderation decisions.

## License and Dataset Attribution

Review the repository's license and the dataset provider's terms before redistributing the code, model weights, or data. The dataset is distributed through the Kaggle competition page linked above; access and usage are subject to its applicable terms.

## Author & Maintainer

**Chaitanya** — Machine Learning and Natural Language Processing

This project explores BERT-based multi-label text classification for automated toxic comment detection, including model training, threshold tuning, evaluation, and local inference.

- **GitHub Profile:** [github.com/chaitu4206](https://github.com/chaitu4206)
- **Project Repository:** [BERT Toxic Content Moderation](https://github.com/chaitu4206/BERT-Toxic-Content-Moderation)

Developed as a machine learning and NLP project focused on responsible automated content moderation.