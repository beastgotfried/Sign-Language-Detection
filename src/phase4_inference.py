import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import time
from collections import deque, Counter
from utils.math_utils import distance, angle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR,"models")
MODEL_PATH= os.path.join(MODELS_DIR,"rf_model.pkl")
ENCODER_PATH=os.path.join(MODELS_DIR,"label_encoder.pkl")

SMOOTH_WINDOW=7
CONF_THRESHOLD=0.6

def load_artifacts(model_path=MODEL_PATH,encoder_path=ENCODER_PATH):
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
    points = []
    features = []

    for i in range(0, len(frame_landmarks), 3):
        points.append([frame_landmarks[i], frame_landmarks[i+1], frame_landmarks[i+2]])

    if len(points) != 21:
        return [0.0] * 23

    wrist = points[0]

    # distances between hands
    features.append(distance(points[4], points[8]))
    features.append(distance(points[8], points[12]))
    features.append(distance(points[12], points[16]))
    features.append(distance(points[16], points[20]))

    # midpoints between the 2 node points
    thumb_mid = np.mean([points[2], points[3]], axis=0)
    index_mid = np.mean([points[6], points[7]], axis=0)
    middle_mid = np.mean([points[10], points[11]], axis=0)
    ring_mid = np.mean([points[14], points[15]], axis=0)
    little_mid = np.mean([points[18], points[19]], axis=0)

    # angles intra finger
    features.append(angle(points[1], thumb_mid, points[4]))
    features.append(angle(points[5], index_mid, points[8]))
    features.append(angle(points[9], middle_mid, points[12]))
    features.append(angle(points[13], ring_mid, points[16]))
    features.append(angle(points[17], little_mid, points[20]))

    # angles btwn finger and wrist
    features.append(angle(wrist, points[1], points[4]))
    features.append(angle(wrist, points[5], points[8]))
    features.append(angle(wrist, points[9], points[12]))
    features.append(angle(wrist, points[13], points[16]))
    features.append(angle(wrist, points[17], points[20]))

    # angles btwn 2 finger
    features.append(angle(points[4], wrist, points[8]))
    features.append(angle(points[8], wrist, points[12]))
    features.append(angle(points[12], wrist, points[16]))
    features.append(angle(points[16], wrist, points[20]))

    # 3 wrist coords
    features.extend([wrist[0], wrist[1], wrist[2]])

    features.extend([0.0, 0.0])

    if len(features) != 23:
        return [0.0] * 23
    return features

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
def main():
    model, encoder = load_artifacts()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    history_labels = deque(maxlen=SMOOTH_WINDOW)
    history_conf = deque(maxlen=SMOOTH_WINDOW)
    prev_t = time.time()

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

                if len(feature_vector) == 23:
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