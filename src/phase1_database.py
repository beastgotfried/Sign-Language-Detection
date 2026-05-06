import cv2
import mediapipe as mp
import os
import argparse
import numpy as np
import pickle


from utils.math_utils import distance,angle #importing distance and angle functions

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=r"D:\Kaggle Dataset\isl_dataset\custom_images"
MODELS_DIR=os.path.join(BASE_DIR,"models")


def parse_args():
    parser= argparse.ArgumentParser(
        description="Extracting hand landmarks from the image",
    )
    parser.add_argument(
        "--dataset-path",
        default=r"D:\Kaggle Dataset\isl_dataset\custom_images",
        help="path to root folder containing data and subfolders",
    )
    parser.add_argument(
        "--output-pickle",
        help="output for the dat extracted from the dataset",
        default=os.path.join(MODELS_DIR,"processed_data.pkl"),
    )
    return parser.parse_args()

def extract_features(frame_landmarks):
    features=[]
    points=[]
    
    for i in range(0,len(frame_landmarks),3):
        points.append([
            frame_landmarks[i],frame_landmarks[i+1],frame_landmarks[i+2],
        ])
    
    if len(points)!=21:
        return [0.0]*23
    
    wrist= points[0]
    
    features.append(distance(points[4],points[8])) #marking the points  
    features.append(distance(points[8],points[12]))
    features.append(distance(points[12],points[16]))
    features.append(distance(points[16],points[20]))
    
    thumb_mid=np.mean([points[2],points[3]],axis=0)
    index_mid=np.mean([points[6],points[7]],axis=0)
    middle_mid=np.mean([points[10],points[11]],axis=0)
    ring_mid=np.mean([points[14],points[15]],axis=0)
    little_mid=np.mean([points[18],points[19]],axis=0)
    
    #angles intra finger
    features.append(angle(points[1],thumb_mid,points[4]))
    features.append(angle(points[5],index_mid,points[8]))
    features.append(angle(points[9],middle_mid,points[12]))
    features.append(angle(points[13],ring_mid,points[16]))
    features.append(angle(points[17],little_mid,points[20]))
    
    
    #angles btwn finger and wrist
    
    features.append(angle(wrist,points[1],points[4]))
    features.append(angle(wrist,points[5],points[8]))
    features.append(angle(wrist,points[9],points[12]))
    features.append(angle(wrist,points[13],points[16]))
    features.append(angle(wrist,points[17],points[20]))
    
    
    #angles btwn 2 finger
    
    features.append(angle(points[4],wrist,points[8]))
    features.append(angle(points[8],wrist,points[12]))
    features.append(angle(points[12],wrist,points[16]))
    features.append(angle(points[16],wrist,points[20]))
    
    features.extend([wrist[0],wrist[1],wrist[2]])
    
    features.extend([0.0,0.0])
    
    return features

def landmarks_to_vector(hand_landmarks):
    frame_data=[]
    for landmark in hand_landmarks.landmark:
        frame_data.extend([landmark.x,landmark.y,landmark.z])
    return frame_data

def main():
    args=parse_args()
    os.makedirs(MODELS_DIR,exist_ok=True)
    
    mp_hands=mp.solutions.hands
    hands= mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.75
    )
    
    data=[]
    labels=[]
    
    print("reading dataset")
    #iterating thru every folder inside the dataset
    for class_folder in os.listdir(args.dataset_path):
        class_path= os.path.join(args.dataset_path,class_folder)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            img_path= os.path.join(class_path,filename)
            if not os.path.exists(img_path):
                print(f"File does not exist: {img_path}")
                continue
            img_bgr=cv2.imread(img_path)
            
            if img_bgr is None:
                print(f"Failed to load image: {img_path}")
                continue
            #scaling image for more accuracy
            max_dimension = 720
            height, width = img_bgr.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img_bgr = cv2.resize(img_bgr,(int(width * scale), int(height * scale)))
                
            img_rgb= cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
            results=hands.process(img_rgb)
            
            if not results.multi_hand_landmarks:
                print(f"No hand detected: {img_path}")
                continue
            
            for hand_landmarks in results.multi_hand_landmarks:
                raw_landmarks= landmarks_to_vector(hand_landmarks)
                features=extract_features(raw_landmarks)
                
                data.append(features)
                labels.append(class_folder)
                
                print(f"processed: {img_path}")
            
    output= {
        "data": data,
        "labels":labels,
        }
    with open(args.output_pickle,"wb") as f:
        pickle.dump(output,f)

    print(f"Saved {len(data)} samples to {args.output_pickle}")

if __name__=="__main__":
    main()

