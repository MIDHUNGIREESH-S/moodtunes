"""Train MoodTune's English emotion model from a public labelled dataset."""

from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def main():
    print("Downloading/loading dair-ai/emotion (English labelled messages)...")
    data = load_dataset("dair-ai/emotion")
    train_text = list(data["train"]["text"]) + list(data["validation"]["text"])
    train_y = list(data["train"]["label"]) + list(data["validation"]["label"])
    test_text = list(data["test"]["text"])
    test_y = list(data["test"]["label"])

    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=100_000
    )
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x_train, train_y)
    predicted = model.predict(x_test)

    print("\nHeld-out test results (measured during this run)")
    print(f"Accuracy: {accuracy_score(test_y, predicted):.4f}")
    print("Precision / recall / F1 by label:")
    print(classification_report(test_y, predicted, target_names=LABEL_NAMES, zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(test_y, predicted))

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(vectorizer, ARTIFACTS / "vectorizer.joblib")
    joblib.dump(model, ARTIFACTS / "classifier.joblib")
    print(f"\nSaved model files to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
