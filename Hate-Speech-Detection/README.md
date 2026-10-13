# Hate Speech Detection Using BERT

A Natural Language Processing (NLP) project that uses a fine-tuned BERT model to classify social media text into three categories: **hate speech, offensive language, and neither**.

**Author:** Chaitanya
**GitHub:** [@chaitu4206](https://github.com/chaitu4206)

The project includes dataset loading, text preprocessing, BERT model training, evaluation, confusion matrix generation, and interactive predictions.

## Project Overview

Hate speech and offensive language on social media can negatively affect online communities. This project applies a pretrained BERT model to classify text and support automated content analysis.

The model performs three-class text classification:

| Class ID | Label | Description |
|---:|---|---|
| 0 | Hate Speech | Text classified as hate speech |
| 1 | Offensive Language | Text classified as offensive language |
| 2 | Neither | Text that belongs to neither of the other categories |

These labels follow the class mapping used by the dataset and project code.

## Features

- BERT-based text classification
- Automatic dataset download when the local dataset is missing
- Text cleaning and normalization
- Stratified dataset sampling and splitting
- Training and evaluation with Hugging Face Transformers
- Classification report generation
- Confusion matrix visualization
- Saved model and tokenizer for local inference
- Interactive command-line predictions with class probabilities

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| BERT | Text classification model |
| Hugging Face Transformers | Tokenization, model training, and inference |
| Hugging Face Datasets | Dataset preparation for evaluation |
| Pandas | Dataset loading and processing |
| NumPy | Numerical operations |
| Scikit-learn | Dataset splitting and classification metrics |
| Matplotlib | Confusion matrix visualization |
| Git and Git LFS | Version control and model weight storage |

## Repository Structure

```text
Hate-Speech-Detection/
├── data/
│   └── hate_speech/
│       └── labeled_data.csv
├── models/
│   └── hate_speech_bert/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── results/
│   └── hate_speech/
│       ├── classification_report.txt
│       └── confusion_matrix.png
├── dataset.py
├── preprocessing.py
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── .gitignore
└── .gitattributes
```

**Note:** The dataset is kept out of Git, while the saved model and selected evaluation artifacts are included in the repository. Model weights are stored using Git LFS.

## Dataset

This project uses the **Hate Speech and Offensive Language** dataset associated with Davidson and collaborators.

- Dataset repository: [Davidson Hate Speech and Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language)
- Dataset file: `labeled_data.csv`

The dataset contains tweets labeled as hate speech, offensive language, or neither.

### Dataset Setup

The project downloads the dataset automatically when it is missing.

Expected local path:

```text
data/hate_speech/labeled_data.csv
```

The loader selects the `tweet` and `class` columns, renames them to `text` and `label`, removes missing values, and converts the labels to integers.

Internet access is required when the dataset is not already available locally.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/chaitu4206/Hate-Speech-Detection.git
cd Hate-Speech-Detection
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the appropriate activation method for your shell or invoke the virtual environment's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The project pins the following dependency versions:

| Package | Version |
|---|---:|
| PyTorch | 2.7.0 |
| Transformers | 5.15.1 |
| Datasets | 5.0.1 |
| Scikit-learn | 1.7.2 |
| Pandas | 2.3.3 |
| NumPy | 2.3.5 |
| Matplotlib | 3.10.6 |

## Text Preprocessing

The preprocessing module performs the following operations:

1. Decodes HTML entities.
2. Replaces Twitter usernames with `@USER`.
3. Replaces URLs with `HTTPURL`.
4. Normalizes repeated whitespace.
5. Tokenizes text using the BERT tokenizer.
6. Truncates or pads token sequences to a maximum length of 128 tokens.

The tokenizer uses the pretrained `bert-base-uncased` model.

Implementation: `preprocessing.py`

## Dataset Preparation

The dataset loader in `dataset.py` provides functions for downloading, loading, sampling, and splitting the data.

The training and evaluation scripts use a sample of up to 10,000 records.

| Subset | Percentage | Samples when 10,000 are available |
|---|---:|---:|
| Training | 80% | 8,000 |
| Validation | 10% | 1,000 |
| Testing | 10% | 1,000 |

Stratified splitting helps preserve the class distribution across subsets. The split uses a fixed random state of 42.

If fewer than 10,000 valid records are available, the actual subset sizes will be smaller.

## Model Training

The project fine-tunes a pretrained `bert-base-uncased` model for three-class sequence classification.

### Training Configuration

| Parameter | Value |
|---|---|
| Pretrained model | `bert-base-uncased` |
| Number of classes | 3 |
| Maximum token length | 128 |
| Training sample limit | 10,000 |
| Training batch size | 8 |
| Evaluation batch size | 8 |
| Number of epochs | 2 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Evaluation strategy | Every epoch |
| Checkpoint saving | Every epoch |
| Best-model metric | Macro F1-score |
| Device configuration | CPU |

### Run Training

From the repository root:

```powershell
python train.py
```

Training requires access to the dataset and pretrained model if they are not already available locally or cached.

The trained model and tokenizer are saved to:

```text
models/hate_speech_bert/
```

The training script uses macro F1-score to select the best model checkpoint.

**Note:** Retraining may produce different model weights and evaluation results.

## Model Evaluation

The evaluation script loads the dataset, creates the configured test split, loads the saved model, and generates predictions.

### Run Evaluation

```powershell
python evaluate.py
```

The script writes outputs to `results/hate_speech/`:

- `classification_report.txt` — precision, recall, F1-score, and support for each class.
- `confusion_matrix.png` — visualization of actual versus predicted classes.
- `test_predictions.csv` — test examples with their true and predicted labels.

The predictions CSV is excluded from Git by the repository's ignore rules.

### Existing Evaluation Results

The existing saved classification report contains results from **1,000 test examples**.

| Metric | Score |
|---|---:|
| Accuracy | 91.30% |
| Macro F1-score | 0.7586 |
| Weighted F1-score | 0.9104 |

### Class-wise Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Hate Speech | 0.4583 | 0.3793 | 0.4151 | 58 |
| Offensive Language | 0.9413 | 0.9522 | 0.9467 | 774 |
| Neither | 0.9112 | 0.9167 | 0.9139 | 168 |

The model performs substantially better on offensive language and neither than on hate speech. The lower F1-score for hate speech indicates that this class is more difficult for the current model to identify reliably.

Overall accuracy should therefore not be interpreted as equally strong performance across all classes.

These metrics describe the existing saved evaluation report. They do not guarantee that retraining or rerunning evaluation will reproduce identical results.

### Understanding the Metrics

- **Precision:** The proportion of predictions for a class that are correct.
- **Recall:** The proportion of actual examples of a class that are correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion matrix:** Shows which classes the model predicts correctly and which classes it confuses.

## Interactive Inference

The inference script loads the saved model and tokenizer from `models/hate_speech_bert/`.

### Run Inference

```powershell
python inference.py
```

Enter a sentence when prompted. The model displays:

- Predicted class
- Confidence for the predicted class
- Probability for hate speech
- Probability for offensive language
- Probability for neither

Type `exit` to stop the interactive session.

### Example Interaction

```text
HATE SPEECH CLASSIFIER

Enter a sentence to classify.
Type 'exit' to stop.

Text: Enter a sentence here

Prediction: OFFENSIVE LANGUAGE
Confidence: 0.8500

Class probabilities:
Hate Speech: 0.1000
Offensive Language: 0.8500
Neither: 0.0500
```

The values above are illustrative only; they are not measured predictions from the saved model.

## Reproducibility

To make experiments easier to reproduce:

1. Install the pinned dependency versions from `requirements.txt`.
2. Use the dataset source configured in `dataset.py`.
3. Keep the configured sample size and random state unchanged.
4. Use the same dataset splitting procedure.
5. Use the saved model and tokenizer from `models/hate_speech_bert/` for inference.
6. Record changes to the training configuration before comparing experiments.

Results may vary if the dataset, model, package versions, hardware, or training configuration changes.

The included model represents a saved model version. Retraining may produce different weights.

## Limitations

- Text classification may miss context, sarcasm, and implicit hate speech.
- The model may confuse offensive language with hate speech.
- Performance can vary across communities, dialects, and topics.
- Model confidence is not a guarantee that a classification is correct.
- The lower hate speech F1-score indicates that this class needs further improvement.
- Automated predictions should be reviewed before being used for high-impact moderation decisions.

## Future Improvements

- Evaluate the model on a separate, representative dataset.
- Improve hate speech detection and investigate class-wise errors.
- Analyze false positives and false negatives.
- Explore class imbalance handling and hyperparameter tuning.
- Add an interactive web interface for predictions.
- Monitor performance across different language varieties and contexts.

## Dataset Attribution

The dataset is associated with Davidson and collaborators' research on hate speech and offensive language on social media.

Consult the [original dataset repository](https://github.com/t-davidson/hate-speech-and-offensive-language) and accompanying research for attribution and usage details.

## Author & Maintainer

**Chaitanya** — Machine Learning and Natural Language Processing

This project explores transformer-based text classification for hate speech and offensive language detection.

- **GitHub Profile:** [github.com/chaitu4206](https://github.com/chaitu4206)
- **Project Repository:** [Hate Speech Detection Using BERT](https://github.com/chaitu4206/Hate-Speech-Detection)

Developed as a machine learning and NLP project focused on responsible text classification and content moderation.
