from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
import image_processing

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "image_features.csv"
MODEL_PATH = ROOT / "models" / "face_model.joblib"
RESULTS_DIR = ROOT / "results"
ID_COLUMNS = ["source_image", "member", "expression", "augmentation"]
THRESHOLD = 0.7
SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_data():
    df = pd.read_csv(CSV_PATH)
    columns = [c for c in df.columns if c not in ID_COLUMNS]
    return df, columns


def make_model():
    return RandomForestClassifier(n_estimators=200, random_state=42)


def known_face_test(df, columns):
    true_all, pred_all, conf_all, blocks = [], [], [], []
    for expression in sorted(df["expression"].unique()):
        train = df[df["expression"] != expression]
        test = df[(df["expression"] == expression) & (df["augmentation"] == "original")]
        model = make_model()
        model.fit(train[columns], train["member"])
        probs = model.predict_proba(test[columns])
        true_all += list(test["member"])
        pred_all += list(model.classes_[probs.argmax(axis=1)])
        conf_all += list(probs.max(axis=1))
        blocks.append(probs)
    classes = model.classes_
    metrics = {
        "accuracy": accuracy_score(true_all, pred_all),
        "f1_score": f1_score(true_all, pred_all, average="macro"),
        "loss": log_loss(true_all, np.vstack(blocks), labels=classes),
    }
    return metrics, np.array(true_all), np.array(pred_all), np.array(conf_all)


def unknown_face_test(df, columns):
    confidences = []
    for member in sorted(df["member"].unique()):
        train = df[df["member"] != member]
        test = df[(df["member"] == member) & (df["augmentation"] == "original")]
        model = make_model()
        model.fit(train[columns], train["member"])
        confidences += list(model.predict_proba(test[columns]).max(axis=1))
    return np.array(confidences)


def error_rates(true, pred, known_conf, unknown_conf, threshold):
    accepted = (known_conf >= threshold) & (pred == true)
    false_reject = 1 - accepted.mean()
    false_accept = (unknown_conf >= threshold).mean()
    return false_reject, false_accept


def train_final(df, columns):
    model = make_model()
    model.fit(df[columns], df["member"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": columns, "threshold": THRESHOLD}, MODEL_PATH)


def identify(image_path):
    bundle = joblib.load(MODEL_PATH)
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, image_processing.SIZE)
    features = image_processing.extract_features(img)
    x = pd.DataFrame([features])[bundle["columns"]]
    probs = bundle["model"].predict_proba(x)[0]
    best = probs.argmax()
    confidence = float(probs[best])
    if confidence >= bundle["threshold"]:
        return bundle["model"].classes_[best], confidence
    return None, confidence


def main():
    df, columns = load_data()
    metrics, true, pred, known_conf = known_face_test(df, columns)
    unknown_conf = unknown_face_test(df, columns)

    print("Known face test (each expression held out in turn)")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 score: {metrics['f1_score']:.3f}")
    print(f"Loss: {metrics['loss']:.3f}")

    rows = []
    for t in SWEEP:
        fr, fa = error_rates(true, pred, known_conf, unknown_conf, t)
        rows.append({"threshold": t, "false_reject_rate": fr, "false_accept_rate": fa})
    sweep = pd.DataFrame(rows)
    print("\nThreshold results (unknown = a member left out of training)")
    print(sweep.round(3).to_string(index=False))

    fr, fa = error_rates(true, pred, known_conf, unknown_conf, THRESHOLD)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{**metrics, "threshold": THRESHOLD, "false_reject_rate": fr, "false_accept_rate": fa}])
    summary.to_csv(RESULTS_DIR / "face_metrics.csv", index=False)
    sweep.to_csv(RESULTS_DIR / "face_threshold_results.csv", index=False)

    train_final(df, columns)
    print(f"\nModel saved to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()