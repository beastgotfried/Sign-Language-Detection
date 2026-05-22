import argparse
import cv2
import mediapipe as mp
import multiprocessing as mp_proc
import numpy as np
import os
import pickle
import time
from collections import deque, Counter
from queue import Queue

from config import (
    BACKSPACE_CONFIDENCE_THRESHOLD,
    BACKSPACE_TOKEN,
    COMMIT_TOKEN,
    CONFIDENCE_THRESHOLD,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    PAUSE_COMMIT_SECONDS,
    PAUSE_CONFIRM_SECONDS,
    STABILITY_FRAMES,
    FEATURE_DIMENSIONS,
)
from phase2_features import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOOTH_WINDOW = STABILITY_FRAMES
CONF_THRESHOLD = CONFIDENCE_THRESHOLD


def parse_args():
    parser = argparse.ArgumentParser(description='Run sign language inference from camera or live webcam.')
    parser.add_argument('--mode', choices=['camera', 'dataset'], default='camera')
    parser.add_argument('--input-pickle', default=os.path.join(BASE_DIR, 'data', 'collected_data.pickle'))
    parser.add_argument('--queue-name', default='sign_predictions', help='Multiprocessing queue name for IPC.')
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


def should_emit_prediction(current_label, current_conf, last_label, last_conf_time):
    """
    Determine if a new stable prediction should be emitted as a token.
    
    Returns (should_emit, token):
    - (True, label) if stable prediction differs from last and confidence is high
    - (False, None) otherwise
    """
    if not current_label or current_conf < CONF_THRESHOLD:
        return False, None
    
    if current_label != last_label:
        return True, current_label
    
    return False, None


def detect_action_hold(stable_label, stable_conf, hold_start_time, action_threshold):
    """
    Detect if a stable prediction has been held long enough to trigger an action.
    
    Returns (should_trigger, seconds_held):
    """
    if not stable_label:
        return False, 0.0
    
    if hold_start_time is None:
        return False, 0.0
    
    seconds_held = time.time() - hold_start_time
    return seconds_held >= action_threshold, seconds_held


def draw_overlay(frame, stable_label, stable_conf, hand_detected, queue_size=0):
    h, w, _ = frame.shape

    base_text = "Hand: YES" if hand_detected else "Hand: NO"
    pred_text = f"Prediction: {stable_label if stable_label else '----'}"
    conf_text = f"Confidence: {stable_conf:.2f}"
    queue_text = f"Queue size: {queue_size}"
    help_text = "Q: Quit | SPACE: Pause"

    cv2.putText(frame, base_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, pred_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, conf_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, queue_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
    cv2.putText(frame, help_text, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    return frame


def run_webcam_inference(model, encoder, output_queue=None):
    """
    Run live webcam inference with queue-based prediction output and pause detection.
    
    Emits to output_queue:
    - Prediction dictionaries with keys: 'label', 'confidence', 'timestamp'
    - Token dictionaries (BACKSPACE_TOKEN, COMMIT_TOKEN) on pause/action events
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    history_labels = deque(maxlen=SMOOTH_WINDOW)
    history_conf = deque(maxlen=SMOOTH_WINDOW)

    last_emitted_label = None
    pause_start_time = None
    hold_start_time = None
    paused = False

    print("Webcam inference running. Press Q to quit, SPACE to pause/resume.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: cannot read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_detected = False
            stable_label = ""
            stable_conf = 0.0

            if results.multi_hand_landmarks and not paused:
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
                    
                    # Check if we should emit this prediction
                    should_emit, token = should_emit_prediction(stable_label, stable_conf, last_emitted_label, None)
                    if should_emit and output_queue is not None:
                        output_queue.put({
                            'type': 'prediction',
                            'label': token,
                            'confidence': stable_conf,
                            'timestamp': time.time()
                        })
                        last_emitted_label = token
                        pause_start_time = time.time()
                        hold_start_time = time.time()
                    
                    # Check for hold-based actions (e.g., backspace after holding)
                    if stable_label and hold_start_time:
                        should_trigger_commit, seconds_held = detect_action_hold(
                            stable_label, stable_conf, hold_start_time, PAUSE_COMMIT_SECONDS
                        )
                        if should_trigger_commit and output_queue is not None:
                            output_queue.put({
                                'type': 'token',
                                'token': COMMIT_TOKEN,
                                'timestamp': time.time()
                            })
                            hold_start_time = None
                            last_emitted_label = None
            else:
                # No hand detected or paused
                if pause_start_time and not paused:
                    seconds_since_last = time.time() - pause_start_time
                    if seconds_since_last >= PAUSE_CONFIRM_SECONDS:
                        if output_queue is not None:
                            output_queue.put({
                                'type': 'token',
                                'token': COMMIT_TOKEN,
                                'timestamp': time.time()
                            })
                        pause_start_time = None
                        last_emitted_label = None

            queue_size = output_queue.qsize() if output_queue else 0
            frame = draw_overlay(frame, stable_label, stable_conf, hand_detected, queue_size)
            cv2.imshow('SignBridge Inference', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"Inference {status}")
                history_labels.clear()
                history_conf.clear()

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam inference stopped")


def main():
    args = parse_args()
    model, encoder = load_artifacts()

    # Create a queue for prediction output (IPC with bridge)
    output_queue = Queue()

    if args.mode == 'camera':
        print("Starting live webcam inference...")
        run_webcam_inference(model, encoder, output_queue)
    else:
        print("Dataset mode not yet implemented in this version.")
        print("Use --mode camera for live inference.")


if __name__ == "__main__":
    main()