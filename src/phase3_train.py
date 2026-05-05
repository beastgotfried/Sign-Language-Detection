import pickle
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = './data'
MODELS = './models'


def parse_args():
    parser = argparse.ArgumentParser(description='Train the sign language classifier from processed feature data.')
    parser.add_argument('--input-pickle', default=os.path.join(DATA_DIR, 'processed_data.pickle'))
    parser.add_argument('--models-dir', default=MODELS)
    return parser.parse_args()


args = parse_args()
input_pickle = args.input_pickle
MODELS = args.models_dir

print(f"Loading processed data from {input_pickle}...")
with open(input_pickle, 'rb') as f:
    dataset = pickle.load(f)
    
x=np.array(dataset['data'])
y_text=np.array(dataset['labels'])

encoder=LabelEncoder()
y_encoded=encoder.fit_transform(y_text)

label_mapping= dict(zip(encoder.classes_,encoder.transform(encoder.classes_)))
print(label_mapping)

x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.2,random_state=42,stratify=y_encoded)

model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

print(f"model accuracy is: {accuracy}")

model_path = os.path.join(MODELS, 'rf_model.pkl')
encoder_path = os.path.join(MODELS, 'label_encoder.pkl')

os.makedirs(MODELS, exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump(model,f)
    
with open(encoder_path, 'wb') as f:
    pickle.dump(encoder,f)
    
print(f"Model data stored to {model_path}")
print(f"Encoder data stored to {encoder_path}")

print(x_train)
print(x_test)
