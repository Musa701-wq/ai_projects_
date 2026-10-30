import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_PATH = os.path.join("csv", "loan_data.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_tree_model.joblib")

FEATURES = ["age", "income", "credit_score", "loan_amount", "employment_years"]


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'.")

    df = pd.read_csv(filepath)
    df = df.dropna()
    return df


def train_decision_tree(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["Approved", "Rejected"])

    return model, acc, report, cm


def save_model(model, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def predict_applicant(applicant_data: dict, model) -> dict:
    sample_df = pd.DataFrame([applicant_data])[FEATURES]
    prediction = model.predict(sample_df)[0]
    probs = model.predict_proba(sample_df)[0]
    classes = model.classes_

    prob_map = {
        cls_name: round(float(prob) * 100, 2)
        for cls_name, prob in zip(classes, probs)
    }

    return {
        "prediction": prediction,
        "confidence": prob_map[prediction],
        "probabilities": prob_map,
    }


def main():
    print("=" * 60)
    print("   LOAN APPROVAL PREDICTION (DECISION TREE MODEL)")
    print("=" * 60)

    print(f"\n[1/4] Loading dataset from '{DATASET_PATH}'...")
    try:
        df = load_data(DATASET_PATH)
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return

    print(f"      Total records: {len(df)}")
    print(f"      Class counts:\n{df['loan_status'].value_counts().to_string()}\n")

    print("[2/4] Training DecisionTreeClassifier (max_depth=3)...")
    model, acc, report, cm = train_decision_tree(df)
    print(f"      Test Accuracy: {acc * 100:.2f}%\n")

    print("--- Classification Report ---")
    print(report)

    print("--- Confusion Matrix ---")
    print(f"Approved (Actual) -> Approved: {cm[0][0]} | Rejected: {cm[0][1]}")
    print(f"Rejected (Actual) -> Approved: {cm[1][0]} | Rejected: {cm[1][1]}\n")

    print(f"[3/4] Saving model to '{MODEL_PATH}'...")
    save_model(model, MODEL_PATH)
    print("      Model saved.\n")

    print("[4/4] Sample Predictions")
    print("-" * 60)
    samples = [
        {"age": 32, "income": 75000, "credit_score": 720, "loan_amount": 20000, "employment_years": 5},
        {"age": 24, "income": 28000, "credit_score": 580, "loan_amount": 35000, "employment_years": 1},
    ]

    for i, s in enumerate(samples, 1):
        res = predict_applicant(s, model)
        print(f"Applicant {i}: {s}")
        print(f"Status:      [ {res['prediction'].upper()} ] (Confidence: {res['confidence']}%)\n")


if __name__ == "__main__":
    main()

