import cv2
import mediapipe as mp
import os
import argparse
import pickle

from phase2_features import extract_features, landmarks_to_vector

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
                features=extract_features(raw_landmarks).tolist()
                
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

