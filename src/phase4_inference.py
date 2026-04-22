import cv2
import mediapipe as mp 
import os 
import pickle     

DATA_DIR= './data'
MODELS='./models'
input_pickle=os.path.join(DATA_DIR,'processed_data.pickle')
#loading the models/artifacts
def load_artifacts():
    with open('rf_model.pkl', "rb") as f:
        model= pickle.load(f) #loading the trained model
    with open('label_encoder.pkl', "rb") as f:
        encoder=pickle.load(f) #loading the encoder that is going to be used to train the model even further
    return model,encoder
    
mp_hands= mp.solutions.hands    
mp_drawing= mp.solutions.drawing_utils   
mp_drawing_styles= mp.solution.drawing_styles

hands= mp.hands.Hands(
    static_image_mode= False,
    max_num_hands= 2,
    min_detection_confidence= 0.3,
    min_tracking_confidence= 0.3
)



