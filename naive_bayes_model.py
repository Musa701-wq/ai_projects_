import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Configure standard output to handle UTF-8 properly across environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_PATH = os.path.join("csv", "spam_dataset.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.joblib")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file and validates the required columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please verify the file path."
        )

    df = pd.read_csv(filepath)

    required_columns = {"label", "message"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    # Remove any missing or null entries
    df = df.dropna(subset=["label", "message"])
    return df


def build_and_train_pipeline(df: pd.DataFrame):
    """
    Splits the dataset, fits a CountVectorizer + MultinomialNB pipeline,
    evaluates classification performance, and returns the trained pipeline.
    """
    X = df["message"]
    y = df["label"]

    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scikit-learn Pipeline combining text extraction and Naive Bayes
    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer(lowercase=True)),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Evaluation on holdout test set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    return pipeline, accuracy, report, cm


def save_model(pipeline: Pipeline, output_path: str):
    """
    Persists the trained model pipeline to disk using joblib.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)


def load_model(model_path: str) -> Pipeline:
    """
    Loads a persisted model pipeline from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'.")
    return joblib.load(model_path)


def predict_text(text: str, pipeline: Pipeline) -> dict:
    """
    Performs inference on a single text string and returns predictions with confidence scores.
    """
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    prob_map = {
        cls_name: round(float(prob) * 100, 2)
        for cls_name, prob in zip(classes, probabilities)
    }

    return {
        "text": text,
        "prediction": prediction,
        "confidence": prob_map[prediction],
        "probabilities": prob_map,
    }


def main():
    print("=" * 65)
    print("   END-TO-END NAIVE BAYES SPAM CLASSIFICATION SYSTEM")
    print("=" * 65)

    # Step 1: Load data from CSV
    print(f"\n[1/4] Loading dataset from '{DATASET_PATH}'...")
    try:
        df = load_data(DATASET_PATH)
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return

    print(f"      Total records loaded: {len(df)}")
    print(f"      Class breakdown:\n{df['label'].value_counts().to_string()}\n")

    # Step 2: Train and evaluate model
    print("[2/4] Training Naive Bayes Pipeline (CountVectorizer + MultinomialNB)...")
    pipeline, accuracy, report, cm = build_and_train_pipeline(df)
    print(f"      Model Accuracy on Test Set: {accuracy * 100:.2f}%\n")

    print("--- Classification Report ---")
    print(report)

    print("--- Confusion Matrix ---")
    print(f"TN (Ham): {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN:       {cm[1][0]} | TP (Spam): {cm[1][1]}\n")

    # Step 3: Save trained pipeline to disk
    print(f"[3/4] Exporting model artifact to '{MODEL_PATH}'...")
    save_model(pipeline, MODEL_PATH)
    print("      Model successfully saved.\n")

    # Step 4: Interactive prediction loop
    print("[4/4] Starting Interactive Inference Engine")
    print("-" * 65)
    print("Type any text message below to classify it in real-time.")
    print("Commands: Type 'exit' or 'quit' to terminate the program.")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nEnter message: ").strip()

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nSession terminated. Program closed.")
                break

            if not user_input:
                print("[!] Please enter non-empty text.")
                continue

            result = predict_text(user_input, pipeline)

            print("\n--- Classification Result ---")
            if result["prediction"] == "spam":
                print("Status:     [ SPAM DETECTED ]")
            else:
                print("Status:     [ LEGITIMATE / HAM MESSAGE ]")

            print(f"Confidence: {result['confidence']}%")
            print(
                f"Probabilities: Ham = {result['probabilities'].get('ham', 0)}% | "
                f"Spam = {result['probabilities'].get('spam', 0)}%"
            )

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user.")
            break


if __name__ == "__main__":
    main()
