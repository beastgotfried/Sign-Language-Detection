import cv2
import mediapipe as mp
import pickle 
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

neon_drawing_style_landmark=mp_drawing_styles.DrawingSpec(
    color=(0,255,136),
    thickness=2,
    circle_radius=4
)
neon_drawing_style_connection=mp_drawing_styles.DrawingSpec(
    color=(0,204,255),
    thickness=3,
    circle_radius=2
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


DATA_DIR='./data/collected_data.pickle'
if not os.path.exists(DATA_DIR):
    os.makedirs('DATA_DIR')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    flip_cam=cv2.flip(frame,1)
    frame.flags.writeable=True
    process_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    process_frame=cv2.flip(process_frame,1)
    results=hands.process(process_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                flip_cam,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                neon_drawing_style_landmark,
                neon_drawing_style_connection
            )
    cv2.imshow('sign collection', flip_cam)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()