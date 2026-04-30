import cv2 #for video capture
import mediapipe as mp #for mediapipe hands and using hand detection
import pickle #storing cooridnates of x y z axis for few command sets
import os #helps store data and route the data into the proper section

mp_hands = mp.solutions.hands #calling the hand position model
mp_drawing = mp.solutions.drawing_utils #drawing utility for marking connections and landmarks
mp_drawing_styles = mp.solutions.drawing_styles #calling the existing and modifiable design styles

dataset_path = "D:\Kaggle Dataset\isl_dataset\custom_images"
images=[]

for filename in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, filename))
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img_rgb)
    
hands= mp_hands.Hands()

#mediapipe only accepts RGB input so we have to change all the input which is in bgr to rgb
for img_rgb in images:
    results= hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)