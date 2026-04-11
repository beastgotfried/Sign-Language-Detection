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
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


DATA_DIR='./data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

labels_to_collect= ["hello","yes","no"]
dataset_size=100
data=[]
labels=[]


cap = cv2.VideoCapture(0)


for label in labels_to_collect:
    print(f"-->Data collection for: {label}")
    
    while True:
        ret, frame = cap.read()
        flip_cam=cv2.flip(frame,1)
        
        cv2.putText(flip_cam,f'please get ready to record "{label}"',(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,136),2)
        cv2.imshow('sign collection',flip_cam)
        
        key=cv2.waitKey(1)
        if key == ord('r'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    counter=0
    while counter<dataset_size:
        ret,frame=cap.read()
        flip_cam=cv2.flip(frame,1)
        frame.flags.writeable = True
        process_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        process_frame=cv2.flip(process_frame, 1)
        results=hands.process(process_frame)

        if results.multi_hand_landmarks:
            frame_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    flip_cam,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    neon_drawing_style_landmark,
                    neon_drawing_style_connection
                )

                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    z=hand_landmarks.landmark[i].z
                    frame_data.extend([x, y, z])

            data.append(frame_data)
            labels.append(label)
            counter+=1

        cv2.putText(flip_cam, f"Collected: {counter}/{dataset_size}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 2)
        cv2.imshow('sign collection',flip_cam)

        key=cv2.waitKey(1)
        if key == ord('r'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            


cap.release()
cv2.destroyAllWindows()

dataset={'data':data,'labels':labels}
pickle_path=os.path.join(DATA_DIR,'collected_data.pickle')

with open(pickle_path,'wb') as f:
    pickle.dump(dataset,f)
    