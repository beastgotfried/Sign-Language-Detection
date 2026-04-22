import pickle
import os
import numpy as np
from utils.math_utils import distance, angle

DATA_DIR='./data'
input_pickle= os.path.join(DATA_DIR, 'collected_data.pickle')
output_pickle= os.path.join(DATA_DIR, 'processed_data.pickle')


print(f"Loading the raw data from {input_pickle}")
with open(input_pickle,'rb') as f:
    dataset=pickle.load(f)
    
    
raw_data=dataset['data']
labels=dataset['labels']
processed_data=[]



def extract_features(frame_landmarks):
    features=[]
    points=[]

    for i in range(0,len(frame_landmarks),3):
        points.append([frame_landmarks[i],frame_landmarks[i+1],frame_landmarks[i+2]])

    if len(points)!=21:
        return [0.0]*23

    wrist= points[0]
    
    features.append(distance(points[4],points[8]))
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

print("extracting features now")
for feature in raw_data:
    extracted_vector=extract_features(feature)
    processed_data.append(extracted_vector)
    
    
processed_data={'data': processed_data, 'labels': labels}

with open(output_pickle, 'wb') as f:
    pickle.dump(processed_data, f)


    
print("Extraction complete")
    
