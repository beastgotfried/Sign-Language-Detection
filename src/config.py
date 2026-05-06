''' config file for the project
all values that will be manually tuned and can be tweaked using input will be present here, i dont have to iterate through any of the code myself, these are like global variables that i am going to define
here which can be used to alter throughout the entire code later on without having to mess with codebase'''

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent
DATA_DIR= BASE_DIR/"data"
MODELS_DIR= BASE_DIR/"models"
LOGS_DIR= BASE_DIR/"logs"
DOCS_DIR=BASE_DIR/"docs"

MODEL_FILENAME="sign_language_model.pkl"
LABEL_ENCODER_FILENAME="label_encoder.pkl"
MODEL_PATH= MODELS_DIR/MODEL_FILENAME
LABEL_ENCODER_PATH=MODELS_DIR/LABEL_ENCODER_FILENAME

#dataset artifacts
RAW_DATA_FILENAME = "collected_data.pickle"
PROCESSED_DATA_FILENAME = "processed_data.pickle"
RAW_DATA_PATH = DATA_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_PATH = DATA_DIR / PROCESSED_DATA_FILENAME


#inference and buffering
CONFIDENCE_THRESHOLD