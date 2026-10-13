# Sensitive Content Detection Using BERT

**Author:** Chaitanya
**GitHub:** [@chaitu4206](https://github.com/chaitu4206)
**Repository:** [Sensitive-Content-Detection](https://github.com/chaitu4206/Sensitive-Content-Detection)

A Natural Language Processing (NLP) project that uses a fine-tuned BERT model to classify text as **sensitive** or **not sensitive**. It uses the X-Sensitive dataset from Hugging Face and provides dataset processing, model training, evaluation, and interactive predictions.

## Project Overview

Sensitive content detection can help identify potentially sensitive text in online platforms and digital communication systems.

This project converts the multi-label annotations in the X-Sensitive dataset into a binary classification task:

| Label ID | Class | Description |
|---:|---|---|
| 0 | Not Sensitive | Text with no sensitive category assigned in the source annotations |
| 1 | Sensitive | Text with at least one sensitive category assigned in the source annotations |

**Important:** These binary labels are derived from the original dataset annotations. This simplified classification task does not identify individual sensitive categories.

## Features

- BERT-based binary text classification
- Dataset loading from Hugging Face
- Conversion of multi-label annotations into binary labels
- CPU-friendly dataset sampling
- Text tokenization and sequence padding
- Fine-tuning with Hugging Face Transformers
- Classification report generation
- Confusion matrix generation
- Saved model and tokenizer for local inference
- Interactive command-line predictions with confidence scores and class probabilities

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| BERT | Pretrained language model |
| Hugging Face Transformers | Model training, tokenization, and inference |
| Hugging Face Datasets | Dataset loading and preparation |
| Scikit-learn | Classification metrics |
| NumPy | Numerical operations |
| Pandas | Data processing dependency |
| Accelerate | Training infrastructure dependency |
| Git and Git LFS | Version control and model weight storage |

## Repository Structure

```text
Sensitive-Content-Detection/
├── models/
│   └── sensitive_bert/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
├── results/
│   └── sensitive/
│       ├── classification_report.txt
│       └── confusion_matrix.txt
├── dataset.py
├── preprocessing.py
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── .gitignore
└── .gitattributes
```

The dataset is downloaded through Hugging Face Datasets rather than stored in the repository. The saved model weights are tracked using Git LFS.

## Dataset

This project uses the [X-Sensitive dataset](https://huggingface.co/datasets/cardiffnlp/x_sensitive), loaded through the Hugging Face Datasets library.

The dataset loader reads the `text` and `labels` fields from each record.

### Binary Label Conversion

The source dataset contains multi-label annotations. This project converts them into binary labels using the following rule:

- If the `labels` list is empty, the text is labeled `0` (not sensitive).
- If the `labels` list contains one or more entries, the text is labeled `1` (sensitive).

The loader processes the available `train`, `validation`, and `test` splits.

It removes empty text entries and limits each split to a maximum of 3,000 records by taking the first records in that split. This is a subset selection, not a randomized sample.

### Dataset Configuration

| Parameter | Value |
|---|---|
| Dataset | `cardiffnlp/x_sensitive` |
| Maximum records per split | 3,000 |
| Classification type | Binary classification |
| Input field | `text` |
| Source annotation field | `labels` |
| Output labels | `0` and `1` |

Internet access is required to download the dataset unless it is already cached locally.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/chaitu4206/Sensitive-Content-Detection.git
cd Sensitive-Content-Detection
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the appropriate activation method for your shell or run the virtual environment's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check dependency compatibility:

```powershell
python -m pip check
```

### Pinned Dependencies

| Package | Version |
|---|---:|
| PyTorch | 2.7.0 |
| Transformers | 5.15.1 |
| Datasets | 5.0.1 |
| Accelerate | 1.14.0 |
| Scikit-learn | 1.7.2 |
| NumPy | 2.3.5 |
| Pandas | 2.3.3 |

The pinned versions document the project's environment. Installation compatibility may depend on the Python version, operating system, and available hardware.

## Text Preprocessing

The preprocessing module uses the pretrained `bert-base-uncased` tokenizer.

| Parameter | Value |
|---|---|
| Pretrained tokenizer | `bert-base-uncased` |
| Maximum sequence length | 128 tokens |
| Truncation | Enabled |
| Padding | `max_length` |

Longer sequences are truncated, and shorter sequences are padded to the configured maximum length.

**Implementation:** `preprocessing.py`

## Model Training

The training script fine-tunes `bert-base-uncased` for two-class sequence classification.

### Training Configuration

| Parameter | Value |
|---|---|
| Pretrained model | `bert-base-uncased` |
| Number of classes | 2 |
| Maximum records per dataset split | 3,000 |
| Maximum token length | 128 |
| Training batch size | 8 |
| Evaluation batch size | 8 |
| Number of epochs | 2 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Output directory | `models/sensitive_bert/` |

### Run Training

From the repository root, execute:

```powershell
python train.py
```

The script loads the dataset, converts the annotations into binary labels, prepares tokenized examples, fine-tunes BERT, and saves the model and tokenizer to:

```text
models/sensitive_bert/
```

The initial pretrained model and tokenizer may need to be downloaded from Hugging Face if they are not already cached. Training time depends on the available hardware and resources.

## Model Evaluation

The evaluation script loads the X-Sensitive dataset, uses the test split, tokenizes the examples, and generates predictions using the saved model.

### Run Evaluation

```powershell
python evaluate.py
```

The evaluation script writes the following files to `results/sensitive/`:

- `classification_report.txt` — precision, recall, F1-score, and support for each class.
- `confusion_matrix.txt` — numerical confusion matrix.

The evaluation script uses the existing saved model in `models/sensitive_bert/`.

## Existing Evaluation Results

The committed classification report contains results for **2,000 test examples**.

| Metric | Score |
|---|---:|
| Accuracy | 78.85% |
| Macro precision | 0.7875 |
| Macro recall | 0.7884 |
| Macro F1-score | 0.7878 |
| Weighted F1-score | 0.7887 |

### Class-wise Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Not Sensitive | 0.7646 | 0.7869 | 0.7756 | 929 |
| Sensitive | 0.8103 | 0.7899 | 0.8000 | 1,071 |

### Confusion Matrix

The existing confusion matrix uses the class order `not_sensitive`, followed by `sensitive`.

| Actual \ Predicted | Not Sensitive | Sensitive |
|---|---:|---:|
| Not Sensitive | 731 | 198 |
| Sensitive | 225 | 846 |

The model correctly classified 731 not-sensitive examples and 846 sensitive examples. It incorrectly classified 198 not-sensitive examples as sensitive and 225 sensitive examples as not sensitive.

These figures describe the saved evaluation artifacts. They do not guarantee that retraining or rerunning evaluation will reproduce identical results.

### Understanding the Metrics

- **Precision:** The proportion of predictions for a class that are correct.
- **Recall:** The proportion of actual examples of a class that are correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Accuracy:** The proportion of all examples classified correctly.
- **Confusion matrix:** Shows correct classifications and errors for each class.

## Interactive Inference

The inference script loads the saved model and tokenizer from `models/sensitive_bert/`.

### Run Inference

```powershell
python inference.py
```

Enter a text sample when prompted. The model returns:

- Predicted label: `SENSITIVE` or `NOT SENSITIVE`
- Confidence score for the predicted class
- Probability for `NOT SENSITIVE`
- Probability for `SENSITIVE`

Type `exit` to stop the interactive session.

### Example Interaction

The following is an illustrative example of the interface, not a measured prediction:

```text
=======================================================
SENSITIVE CONTENT CLASSIFIER
=======================================================

Type 'exit' to stop.

Text: Enter a text sample here

Prediction: SENSITIVE
Confidence: 0.8500

Class probabilities:
NOT SENSITIVE: 0.1500
SENSITIVE: 0.8500
```

The probabilities in this example are placeholders. Actual values depend on the input and saved model.

## Reproducibility

To make experiments easier to reproduce:

1. Install the pinned dependencies from `requirements.txt`.
2. Use the dataset identifier configured in `dataset.py`.
3. Keep the dataset split and first-record sampling procedure unchanged.
4. Use the same preprocessing and maximum token length.
5. Use the saved model and tokenizer in `models/sensitive_bert/` for inference.
6. Record changes to the dataset, model configuration, and training parameters before comparing results.

Results may vary if the dataset version, model weights, package versions, or training configuration changes.

## Limitations

- The project reduces multi-label annotations to a single binary label, losing information about individual sensitive categories.
- The binary rule treats any nonempty annotation list as sensitive.
- The model may miss context, implicit sensitivity, sarcasm, or ambiguous language.
- The saved evaluation results show that some sensitive and not-sensitive examples are misclassified.
- Model confidence is not a guarantee that a prediction is correct.
- The evaluation report alone does not establish performance across every type of sensitive content or real-world population.
- Predictions should be reviewed before being used in high-impact moderation decisions.

## Future Improvements

- Preserve and predict individual sensitive categories instead of collapsing them into binary labels.
- Evaluate on additional representative datasets.
- Analyze false positives and false negatives.
- Investigate class imbalance and threshold selection.
- Add a web interface for interactive predictions.
- Test performance across different text types, communities, and language varieties.

## Author and Maintainer

Developed by **Chaitanya** as a machine learning and Natural Language Processing project for binary sensitive-content classification.

- GitHub profile: https://github.com/chaitu4206
- Repository: https://github.com/chaitu4206/Sensitive-Content-Detection
