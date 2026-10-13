from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocessing import LABELS


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final"

THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "evaluation-v3" / "best_thresholds.npy"
)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# VALIDATE MODEL PATH
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_PATH}"
    )

if not (MODEL_PATH / "config.json").exists():
    raise FileNotFoundError(
        f"config.json not found inside: {MODEL_PATH}"
    )

if not (MODEL_PATH / "tokenizer.json").exists():
    raise FileNotFoundError(
        f"tokenizer.json not found inside: {MODEL_PATH}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    use_fast=True,
    local_files_only=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading classification model...")

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model.to(device)
model.eval()

print("Model loaded successfully!")


# ============================================================
# LOAD CLASSIFICATION THRESHOLDS
# ============================================================

thresholds = {label: 0.5 for label in LABELS}

if THRESHOLD_PATH.exists():
    try:
        saved_thresholds = np.load(
            str(THRESHOLD_PATH),
            allow_pickle=True
        ).item()

        if isinstance(saved_thresholds, dict):
            for label in LABELS:
                if label in saved_thresholds:
                    value = float(saved_thresholds[label])

                    if 0.0 <= value <= 1.0:
                        thresholds[label] = value

        print("Saved classification thresholds loaded.")

    except (ValueError, TypeError, OSError, AttributeError) as error:
        print(f"Could not load thresholds: {error}")
        print("Using default threshold of 0.5.")

else:
    print("Threshold file not found. Using default threshold of 0.5.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_toxicity(text):
    """
    Predict toxicity labels for the supplied text.

    Returns a dictionary containing each label's probability,
    threshold, and predicted classification.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide non-empty text.")

    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    encoded_input = {
        key: value.to(device)
        for key, value in encoded_input.items()
    }

    with torch.no_grad():
        outputs = model(**encoded_input)
        probabilities = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    results = {}

    for index, label in enumerate(LABELS):
        probability = float(probabilities[index])
        threshold = thresholds[label]

        results[label] = {
            "probability": probability,
            "threshold": threshold,
            "prediction": probability >= threshold
        }

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(text):
    results = predict_toxicity(text)

    print("\n" + "=" * 60)
    print("TEXT:")
    print(text)
    print("=" * 60)

    detected_labels = []

    for label, details in results.items():
        probability = details["probability"]
        threshold = details["threshold"]
        prediction = details["prediction"]

        status = "DETECTED" if prediction else "NOT DETECTED"

        print(f"\nLabel: {label}")
        print(f"Probability: {probability:.4f} ({probability * 100:.2f}%)")
        print(f"Threshold:   {threshold:.4f}")
        print(f"Prediction:  {status}")

        if prediction:
            detected_labels.append(label)

    print("\n" + "-" * 60)

    if detected_labels:
        print("Potentially toxic content detected.")
        print("Detected labels:", ", ".join(detected_labels))
    else:
        print("No toxicity labels crossed their thresholds.")

    print("=" * 60)


# ============================================================
# INTERACTIVE PREDICTION LOOP
# ============================================================

if __name__ == "__main__":
    print("\nBERT Toxic Content Detection")
    print("Enter text to analyze.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_text = input("Enter text: ").strip()

            if user_text.lower() in {"exit", "quit"}:
                print("Exiting inference program.")
                break

            if not user_text:
                print("Please enter some text.\n")
                continue

            display_results(user_text)

        except KeyboardInterrupt:
            print("\nExiting inference program.")
            break

        except (ValueError, RuntimeError) as error:
            print(f"\nPrediction error: {error}")