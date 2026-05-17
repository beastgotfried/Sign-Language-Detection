import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import argparse
from collections import deque, Counter
from config import CONFIDENCE_THRESHOLD, LABEL_ENCODER_PATH, MODEL_PATH, STABILITY_FRAMES, FEATURE_DIMENSIONS
from phase2_features import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOOTH_WINDOW = STABILITY_FRAMES
CONF_THRESHOLD = CONFIDENCE_THRESHOLD


def parse_args():
    parser = argparse.ArgumentParser(description='Run sign language inference from camera or dataset input.')
    parser.add_argument('--mode', choices=['camera', 'dataset'], default='dataset')
    parser.add_argument('--input-pickle', default=os.path.join(BASE_DIR, 'data', 'collected_data.pickle'))
    return parser.parse_args()

def load_artifacts(model_path=str(MODEL_PATH), encoder_path=str(LABEL_ENCODER_PATH)):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return model, encoder

def extract_features_live(frame_landmarks):
    return extract_features(frame_landmarks)

def predict_sign(model, encoder, feature_vector):
    x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    pred_id = model.predict(x)[0]
    pred_label = encoder.inverse_transform([pred_id])[0]
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        confidence = float(np.max(probs))
    return pred_label, confidence


def smooth_prediction(history_labels, history_conf, current_label, current_conf):
    history_labels.append(current_label)
    history_conf.append(current_conf)

    if not history_labels:
        return "", 0.0

    label_counts = Counter(history_labels)
    stable_label = label_counts.most_common(1)[0][0]

    stable_conf_vals = [c for l, c in zip(history_labels, history_conf) if l == stable_label]
    stable_conf = float(np.mean(stable_conf_vals)) if stable_conf_vals else 0.0

    if stable_conf < CONF_THRESHOLD:
        return "", stable_conf
    return stable_label, stable_conf
def draw_overlay(frame, stable_label, stable_conf, hand_detected):
    h, w, _ = frame.shape

    base_text = "Welcome" if hand_detected else "Hand: NO"
    pred_text = f"Prediction: {stable_label if stable_label else '....'}"
    conf_text = f"Confidence: {stable_conf:.2f}"
    help_text = "Q: Quit"

    cv2.putText(frame, base_text, (20, 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, pred_text, (20, 65), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, conf_text, (20, 95), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, help_text, (20, 115), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def run_dataset_inference(model, encoder, input_pickle):
    if not os.path.exists(input_pickle):
        raise FileNotFoundError(f"Dataset file not found: {input_pickle}")

    with open(input_pickle, 'rb') as f:
        dataset = pickle.load(f)

    raw_data = dataset.get('data', [])
    labels = dataset.get('labels', [])

    print(f"Loaded {len(raw_data)} samples from {input_pickle}")

    correct = 0
    total = 0

    for index, frame_landmarks in enumerate(raw_data):
        feature_vector = extract_features_live(frame_landmarks)
        if len(feature_vector) != FEATURE_DIMENSIONS:
            continue

        predicted_label, confidence = predict_sign(model, encoder, feature_vector)
        actual_label = labels[index] if index < len(labels) else None

        if actual_label is not None:
            total += 1
            if predicted_label == actual_label:
                correct += 1

        if actual_label is not None:
            print(f"Sample {index + 1}: actual={actual_label} predicted={predicted_label} confidence={confidence:.2f}")
        else:
            print(f"Sample {index + 1}: predicted={predicted_label} confidence={confidence:.2f}")

    if total:
        print(f"Dataset accuracy: {correct / total:.4f} ({correct}/{total})")


def main():
    args = parse_args()
    model, encoder = load_artifacts()

    if args.mode == 'dataset':
        run_dataset_inference(model, encoder, args.input_pickle)
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    history_labels = deque(maxlen=SMOOTH_WINDOW)
    history_conf = deque(maxlen=SMOOTH_WINDOW)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_detected = False
            stable_label = ""
            stable_conf = 0.0

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                frame_data = []
                for lm in hand_landmarks.landmark:
                    frame_data.extend([lm.x, lm.y, lm.z])

                feature_vector = extract_features_live(frame_data)

                if len(feature_vector) == FEATURE_DIMENSIONS:
                    current_label, current_conf = predict_sign(model, encoder, feature_vector)
                    stable_label, stable_conf = smooth_prediction(
                        history_labels, history_conf, current_label, current_conf
                    )

            frame = draw_overlay(frame, stable_label, stable_conf, hand_detected)
            cv2.imshow('Sign Language', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()