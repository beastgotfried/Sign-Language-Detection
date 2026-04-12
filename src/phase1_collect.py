#importing libraries
import cv2 #for video capture
import mediapipe as mp #for mediapipe hands and using hand detection
import pickle #storing cooridnates of x y z axis for few command sets
import os #helps store data and route the data into the proper section

mp_hands = mp.solutions.hands #calling the hand position model
mp_drawing = mp.solutions.drawing_utils #drawing utility for marking connections and landmarks
mp_drawing_styles = mp.solutions.drawing_styles #calling the existing and modifiable design styles

neon_drawing_style_landmark=mp_drawing_styles.DrawingSpec( #creating a design style for ladnmarks
    color=(0,255,136),
    thickness=2,
    circle_radius=4
)
neon_drawing_style_connection=mp_drawing_styles.DrawingSpec( #creating a design style for connections
    color=(0,204,255),
    thickness=3,
    circle_radius=2
)

hands = mp_hands.Hands( #calling the hand detection model and giving conditions 
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3, #confidence of model to detect hand
    min_tracking_confidence=0.3
)


DATA_DIR='./data' #routing data directory to data folder
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

labels_to_collect= ["hello","yes","no"] #Creating symbols to capture for gestures
dataset_size=100 #number of frames to capture per gesture
data=[] #initializing empty array for data
labels=[] #initializing empty array for completed labels


cap = cv2.VideoCapture(0) #initializing video capture


for label in labels_to_collect: #creating loops to collect data
    print(f"-->Data collection for: {label}")
    
    while True:
        ret, frame = cap.read()
        flip_cam=cv2.flip(frame,1)
        #writing on the camera
        cv2.putText(flip_cam,f'please get ready to record "{label}"',(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,136),2)
        cv2.imshow('sign collection',flip_cam)
        #breaking loop using force and starting recording
        key=cv2.waitKey(1)
        if key == ord('r'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    #comes to here if we press r and starts recording
    counter=0 #counts number of frames
    
    #storing of data
    while counter<dataset_size:
        ret,frame=cap.read()
        flip_cam=cv2.flip(frame,1)
        frame.flags.writeable = True
        process_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#mediapipe only accepts rgb input so converting opencv input to rgb
        process_frame=cv2.flip(process_frame, 1) #flipping input as per need 
        results=hands.process(process_frame) #running mediapipe model through video input
        #marking connections and landmarks
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
                    #running math to analyse position of hand and converting to xyz values to store data in pickle file
                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    z=hand_landmarks.landmark[i].z
                    frame_data.extend([x, y, z])
            #appending data to the pickle file in the dataset
            data.append(frame_data)
            labels.append(label)
            counter+=1
        #showing the number of frames collected
        cv2.putText(flip_cam, f"Collected: {counter}/{dataset_size}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 2)
        cv2.imshow('sign collection',flip_cam)
        #i dont know why this is here, i think i am too retarded. there is no need for this to be here we already added a force quit section
        key=cv2.waitKey(1)
        if key == ord('r'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            


#another retarded moment of mine
cap.release()
cv2.destroyAllWindows()
#sending data into the pickel file
dataset={'data':data,'labels':labels}
pickle_path=os.path.join(DATA_DIR,'collected_data.pickle')
#dumping dataset into pickle file
with open(pickle_path,'wb') as f:
    pickle.dump(dataset,f)
    