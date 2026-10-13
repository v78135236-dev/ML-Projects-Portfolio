# Violence Detection Using BERT

A binary text-classification project that fine-tunes **BERT (`bert-base-uncased`)** to classify text as **VIOLENCE** or **NOT VIOLENCE**. The project includes dataset preparation, text preprocessing, model training, evaluation, and interactive prediction.

**Author:** Chaitanya
**GitHub:** [@chaitu4206](https://github.com/chaitu4206)
**Repository:** [Violence-Detection](https://github.com/chaitu4206/Violence-Detection)

## Overview

Online platforms need ways to identify potentially violent language. This project fine-tunes a pretrained BERT transformer model for binary text classification using the `kcrl/Violence` dataset from Hugging Face.

### Features

- Download and prepare the dataset using Hugging Face Datasets
- Convert three original label categories into a binary classification task
- Remove missing records and duplicate text entries
- Clean URLs and normalize whitespace
- Tokenize text using the BERT tokenizer
- Fine-tune BERT for sequence classification
- Evaluate using precision, recall, F1-score, accuracy, and a confusion matrix
- Perform interactive command-line predictions
- Track trained model weights using Git Large File Storage (Git LFS)

## Model Performance

The following results are taken from the saved evaluation report for **500 test examples**.

| Metric | NOT VIOLENCE | VIOLENCE |
|---|---:|---:|
| Precision | 66.01% | 67.53% |
| Recall | 76.23% | 55.74% |
| F1-score | 70.75% | 61.07% |
| Test support | 265 | 235 |

### Overall Metrics

| Metric | Result |
|---|---:|
| Accuracy | 66.60% |
| Macro precision | 66.77% |
| Macro recall | 65.99% |
| Macro F1-score | 65.91% |
| Weighted F1-score | 66.20% |

The model correctly classified **333 out of 500** test examples.

### Confusion Matrix

Class order: `0 = NOT VIOLENCE`, `1 = VIOLENCE`.

| Actual / Predicted | NOT VIOLENCE | VIOLENCE |
|---|---:|---:|
| NOT VIOLENCE | 202 | 63 |
| VIOLENCE | 104 | 131 |

![Violence classification confusion matrix](results/violence/confusion_matrix.png)

The model misclassified 104 of the 235 violent examples as non-violent. Its violence-class recall is 55.74%, indicating that it misses a substantial proportion of positive examples. These results show that further improvement and validation are necessary before consequential moderation use.

## Technology Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Transformer model:** Hugging Face Transformers (BERT)
- **Dataset handling:** Hugging Face Datasets
- **Data processing:** Pandas, NumPy
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib
- **Version control and model storage:** Git, Git LFS

## Project Structure

```text
Violence-Detection/
├── dataset.py
├── download_dataset.py
├── preprocessing.py
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── .gitignore
├── .gitattributes
├── models/
│   └── violence_bert/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
└── results/
    └── violence/
        ├── classification_report.txt
        └── confusion_matrix.png
```

The `data/` directory is generated locally and contains the downloaded dataset and prepared training, validation, and test CSV files. The generated `test_predictions.csv` is excluded from version control.

## Dataset

This project uses the [`kcrl/Violence`](https://huggingface.co/datasets/kcrl/Violence) dataset hosted on Hugging Face.

Download and prepare the source data:

```bash
python download_dataset.py
```

The download script loads the available dataset splits, selects the `text` and `label` columns, combines the splits, removes missing values and duplicate text, and saves the prepared source data to:

```text
data/violence/violence_data.csv
```

### Label Mapping

The original three-class labels are converted into a binary classification task.

| Original label | Final label | Meaning |
|---|---:|---|
| 0 — Non-Violence | 0 | NOT VIOLENCE |
| 1 — Passive Violence | 1 | VIOLENCE |
| 2 — Direct Violence | 1 | VIOLENCE |

Both passive and direct violence are grouped into the same positive class.

### Dataset Preparation and Splitting

The `dataset.py` script:

1. Loads `data/violence/violence_data.csv`.
2. Checks for the required `text` and `label` columns.
3. Removes missing values and retains labels 0 and 1.
4. Converts text to strings and labels to integers.
5. Shuffles the data using random state 42.
6. Limits the working dataset to a maximum of 5,000 samples.
7. Creates stratified training, validation, and test splits.

The intended split is:

| Split | Proportion | Approximate size with 5,000 records |
|---|---:|---:|
| Training | 80% | 4,000 |
| Validation | 10% | 500 |
| Testing | 10% | 500 |

Actual sizes depend on the number of usable records. The generated CSV files are stored in `data/violence/`:

- `train.csv`
- `validation.csv`
- `test.csv`

## Text Preprocessing

The `preprocessing.py` module provides text-cleaning and CSV-loading functions.

The `clean_text()` function:

- Converts the input to a string
- Removes URL patterns beginning with `http` or `www`
- Replaces repeated whitespace with a single space
- Removes leading and trailing whitespace

The BERT tokenizer pads and truncates sequences to a maximum length of **64 tokens**.

## Installation

### Prerequisites

- Python compatible with the pinned package versions
- Git
- Git LFS
- Internet access for downloading the dataset and any required pretrained model files

### 1. Clone the Repository

```bash
git clone https://github.com/chaitu4206/Violence-Detection.git
cd Violence-Detection
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project specifies these dependency versions:

```text
torch==2.7.0
transformers==5.15.1
datasets==5.0.1
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.7.2
matplotlib==3.10.6
accelerate==1.14.0
```

### 4. Retrieve the Trained Model

The trained model weights are tracked using Git LFS. Install Git LFS and retrieve the files:

```bash
git lfs install
git lfs pull
```

Verify that `models/violence_bert/model.safetensors` contains the actual model weights, rather than an LFS pointer.

## Training

First, download and prepare the dataset:

```bash
python download_dataset.py
```

Then train the model:

```bash
python train.py
```

The training pipeline prepares the dataset splits, tokenizes the text, fine-tunes BERT, and saves the model and tokenizer.

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Maximum dataset samples | 5,000 |
| Number of classes | 2 |
| Maximum sequence length | 64 tokens |
| Training batch size | 8 |
| Evaluation batch size | 8 |
| Epochs | 2 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Best-model selection | Lowest evaluation loss |
| Checkpoint retention | 1 |

The trained model and tokenizer are saved to:

```text
models/violence_bert/
```

Training may take considerable time, especially on a CPU.

## Evaluation

Run the evaluation script after the dataset splits and trained model are available:

```bash
python evaluate.py
```

The script loads `data/violence/test.csv`, tokenizes the test examples, loads the trained model, generates predictions, and calculates classification metrics.

| Output | Purpose |
|---|---|
| `results/violence/classification_report.txt` | Precision, recall, F1-score, support, and accuracy |
| `results/violence/confusion_matrix.png` | Visualization of actual versus predicted labels |
| `results/violence/test_predictions.csv` | Test records with predicted labels and class names |

The prediction CSV is excluded from Git through `.gitignore`.

**Reproducibility note:** The published metrics above come from the saved report. Retraining the model or changing the dataset may produce different results. Regenerate the report before updating the performance figures.

## Interactive Inference

Start the interactive classifier:

```bash
python inference.py
```

Enter text at the prompt. Type `exit` to stop.

Example session — illustrative output only:

```text
=======================================================
VIOLENCE CLASSIFIER
=======================================================

Enter text to classify.
Type 'exit' to stop.

Text: Example text to classify

Prediction: NOT VIOLENCE
Confidence: 0.XXXX

Class probabilities:
NOT VIOLENCE: 0.XXXX
VIOLENCE: 0.XXXX
```

The output depends on the input and trained model. This example does not represent a verified prediction.

### Use the Prediction Function in Python

```python
from inference import predict

text = "Example text to classify"

predicted_class, confidence, probabilities = predict(text)

label = "VIOLENCE" if predicted_class == 1 else "NOT VIOLENCE"

print("Prediction:", label)
print("Confidence:", round(confidence, 4))
print("Probabilities:", probabilities)
```

The `predict()` function returns:

- Predicted class ID (`0` or `1`)
- Predicted class confidence
- Softmax probabilities ordered as NOT VIOLENCE, then VIOLENCE

Confidence is a model-generated probability, not a guarantee that the prediction is correct.

## Limitations and Responsible Use

- The saved evaluation accuracy is 66.60%; the model is not sufficiently validated for high-stakes decisions.
- Violence-class recall is 55.74%, so the model misses a meaningful proportion of violent examples.
- Fine-tuning uses at most 5,000 records, limiting the amount of training data.
- Dataset labels and source distribution may not represent every form of violent language, context, dialect, or language.
- Quotation, satire, fictional dialogue, and discussion of violence can be difficult to classify from text alone.
- The model should not be treated as a definitive judge of a person's intent or used as the sole basis for content removal or disciplinary action.

For practical moderation, evaluate the model on representative, independently collected data and include appropriate human review and appeal processes.

## Future Improvements

- Increase and diversify the training dataset
- Improve recall for the violence class
- Compare against simpler baseline models
- Tune decision thresholds using validation data
- Evaluate performance across different text sources and language varieties
- Add automated tests for preprocessing and inference
- Build an API or interface for interactive demonstrations
- Track model versions and evaluation metrics across experiments

## Author and Repository

**Author:** Chaitanya

- GitHub profile: https://github.com/chaitu4206
- Project repository: https://github.com/chaitu4206/Violence-Detection

## License

No license file was identified in the inspected project files. Add a `LICENSE` file before distributing the project under a specific open-source license.
