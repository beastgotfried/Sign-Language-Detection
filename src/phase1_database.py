import cv2
import mediapipe as mp
import os



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

dataset_path = r"D:\Kaggle Dataset\isl_dataset\custom_images"
images = []

for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f" Failed to load: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images.append((class_folder, img_bgr, img_rgb))
        print(f"Loaded: {img_path} | Shape: {img_bgr.shape}")

# Initialize MediaPipe Hands
hands = mp_hands.Hands()

# Process images
for class_label, img_bgr, img_rgb in images:
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Show the annotated image
    cv2.imshow(f"Class: {class_label}", img_bgr)
    cv2.waitKey(0)

cv2.destroyAllWindows()
