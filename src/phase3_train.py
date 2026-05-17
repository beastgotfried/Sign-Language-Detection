import argparse
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    LABEL_ENCODER_FILENAME,
    MIN_CLASS_ACCURACY,
    MODEL_FILENAME,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
    TEST_ACCURACY_TARGET,
    TRAIN_SPLIT_RATIO,
)

DATA_DIR = "./data"
MODELS = "./models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sign language classifier from processed feature data.")
    parser.add_argument("--input-pickle", default=os.path.join(DATA_DIR, "processed_data.pickle"))
    parser.add_argument("--models-dir", default=MODELS)
    return parser.parse_args()


def _class_accuracy_table(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, encoder: LabelEncoder) -> list[tuple[str, float, int]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    table: list[tuple[str, float, int]] = []
    for label_idx in labels:
        class_name = encoder.inverse_transform([label_idx])[0]
        row_sum = int(matrix[label_idx, :].sum())
        class_acc = 0.0 if row_sum == 0 else float(matrix[label_idx, label_idx]) / float(row_sum)
        table.append((class_name, class_acc, row_sum))
    return table


def main() -> None:
    args = parse_args()
    input_pickle = args.input_pickle
    models_dir = args.models_dir

    print(f"Loading processed data from {input_pickle}...")
    with open(input_pickle, "rb") as f:
        dataset = pickle.load(f)

    x = np.array(dataset["data"])
    y_text = np.array(dataset["labels"])

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_text)

    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("Label mapping:")
    print(label_mapping)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=TRAIN_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )

    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")
    if test_accuracy < TEST_ACCURACY_TARGET:
        print(f"Warning: test accuracy below target {TEST_ACCURACY_TARGET:.2f}")

    labels = np.unique(y_encoded)
    class_table = _class_accuracy_table(y_test, test_pred, labels, encoder)

    print("Per-class accuracy (test split):")
    for class_name, class_acc, sample_count in class_table:
        status = "OK" if class_acc >= MIN_CLASS_ACCURACY else "LOW"
        print(f"- {class_name}: {class_acc:.4f} ({sample_count} samples) [{status}]")

    weak_classes = [name for name, acc, _ in class_table if acc < MIN_CLASS_ACCURACY]
    if weak_classes:
        print("Classes below minimum threshold:")
        for class_name in weak_classes:
            print(f"- {class_name}")

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, MODEL_FILENAME)
    encoder_path = os.path.join(models_dir, LABEL_ENCODER_FILENAME)
    cm_npy_path = os.path.join(models_dir, "confusion_matrix.npy")
    cm_csv_path = os.path.join(models_dir, "confusion_matrix.csv")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    cm = confusion_matrix(y_test, test_pred, labels=labels)
    np.save(cm_npy_path, cm)
    np.savetxt(cm_csv_path, cm, fmt="%d", delimiter=",")

    print(f"Model stored to {model_path}")
    print(f"Encoder stored to {encoder_path}")
    print(f"Confusion matrix stored to {cm_npy_path} and {cm_csv_path}")


if __name__ == "__main__":
    main()
