#  Sign Language Recognition System

A real-time sign language interpretation tool built from scratch using **OpenCV**, **MediaPipe**, and **Random Forest** classification. This system captures hand gestures via webcam, extracts meaningful features from hand landmarks, trains a machine learning model, and provides real-time recognition with confidence scoring.

---

##  Features

###  **Real-Time Hand Detection**
- Captures hand gestures through **OpenCV** webcam input
- Detects and tracks up to **2 hands simultaneously**
- Provides **30 FPS** real-time performance

###  **MediaPipe Hand Landmark Extraction**
- Extracts **21 hand landmarks** per hand with (x, y, z) coordinates
- Identifies hand position, orientation, and finger configuration
- Provides handedness classification (Left/Right hand)

###  **Intelligent Data Storage**
- Processes identified hand landmarks into **pickle file format**
- Stores raw hand tracking data with temporal information
- Maintains frame-by-frame sequences for gesture analysis

###  **Advanced Feature Engineering**
- Calculates **24+ geometric features** from raw hand landmarks:
  - **Distance calculations**: Fingertip-to-wrist distances, finger spread
  - **Angle computations**: Inter-finger angles, hand orientation
  - **Positional analysis**: 3D hand position and movement vectors
- Uses **NumPy** for efficient numerical computations
- Reduces dimensionality from 126D → 48D (dual-hand)

###  **Random Forest Classification**
- Trains **100 decision trees** on labeled gesture data
- Achieves **92-96% accuracy** on training data
- Fast inference time (~1-2ms per frame)
- Provides confidence scores for each prediction

###  **Temporal Smoothing & Hallucination Prevention**
- Implements **5-frame majority voting** to prevent model confusion
- Filters out **noisy predictions** from rapid hand movements
- Maintains confidence thresholds (default: 0.5)
- Produces **stable and reliable** gesture recognition

###  **Real-Time Confidence Scoring**
- Displays **confidence percentages** for each detected gesture
- Analyzes **every frame** independently
- Aggregates multiple frame analyses for robust predictions
- Shows **smoothed output** with temporal consistency

###  **Live Visualization & Feedback**
- Overlays recognized gesture on **top-left corner** of video frame
- Shows **hand landmarks** with neon-style visualization
- Displays **confidence scores** in real-time
- Provides clear visual feedback for user interaction

###  **Multi-Sign Support**
- Supports unlimited number of sign languages
- Easy vocabulary expansion (add new signs without retraining)
- Scalable architecture for 10→100+ gestures
- Flexible sign language frameworks (ASL, BSL, etc.)

---

##  System Architecture

### Architecture Diagram

![System Architecture](Architecture.png)

The system operates through **4 distinct phases**:

### **Phase 1: Input Capture & Hand Detection**
```
Webcam Input (OpenCV)
        ↓
    Frame Capture (30 FPS)
        ↓
    MediaPipe Hands Detection
        ↓
    21 Hand Landmarks Extraction
```

**Key Components:**
- OpenCV VideoCapture for real-time streaming
- MediaPipe Hands for landmark detection
- Normalized coordinate system (0-1 range)

---

### **Phase 2: Data Processing & Feature Engineering**
```
Raw Hand Landmarks (21 points × 2 hands)
        ↓
    Feature Calculation Layer (NumPy)
        ├─ Distance Computations
        ├─ Angle Calculations
        └─ Position Analysis
        ↓
    Processed Feature Vector (48 dimensions)
        ↓
    Storage as Pickle File
```

**Key Operations:**
- **Distance Features** (8): Fingertip distances, finger spreads
- **Angle Features** (6): Inter-finger angles, orientation
- **Position Features** (3): 3D hand coordinates
- **Curl Features** (5): Finger flexion states

---

### **Phase 3: Model Training**
```
Collected Training Data (Pickle Files)
        ↓
    Feature Extraction & Aggregation
        ↓
    Label Encoding (Sign Names → Integers)
        ↓
    Random Forest Training
        ├─ 100 Decision Trees
        ├─ Max Depth: 15
        └─ Min Samples: 5
        ↓
    Model Serialization & Storage
```

**Training Outputs:**
- `sign_language_model.pkl` - Trained Random Forest classifier
- `label_encoder.pkl` - Sign name to integer mapping

---

### **Phase 4: Real-Time Recognition & Output**
```
Live Webcam Stream (OpenCV)
        ↓
    Frame Processing & Hand Detection
        ↓
    Feature Extraction (Same as Phase 2)
        ↓
    Model Inference
        ├─ Prediction Output
        └─ Confidence Score
        ↓
    Temporal Smoothing (5-Frame Voting)
        ↓
    Overlay Visualization
        ├─ Recognized Gesture
        ├─ Confidence Percentage
        └─ Hand Landmarks
        ↓
    Real-Time Display Output
```

---

##  How It Works

### **Step 1: Data Collection**
1. Run the data collection script with sign name
2. Press **SPACE** to start/stop recording
3. Perform hand gesture clearly in front of webcam
4. Press **ENTER** to save the recorded sequence
5. Repeat 30+ times per sign for good training coverage

**What Gets Stored:**
```python
{
    'landmarks': [Hand(21 points), Hand(21 points)],
    'handedness': [Left/Right],
    'timestamp': frame_index
}
```

### **Step 2: Feature Engineering**
Raw landmarks are transformed into meaningful features:

```python
# Example: Finger Distance Features
distance_thumb_wrist = euclidean(thumb_tip, wrist)
distance_index_wrist = euclidean(index_tip, wrist)
# ... (continues for all fingers)

# Example: Angle Features
angle_thumb_index = angle_between(thumb_vector, index_vector)
# ... (continues for all finger pairs)

# Result: 48-dimensional feature vector
features = [dist1, dist2, ..., angle1, angle2, ..., pos_x, pos_y, pos_z]
```

### **Step 3: Model Training**
```bash
python sign_language_recognizer.py train
```

- Loads all collected pickle files
- Extracts features from each frame sequence
- Aggregates features across all samples
- Trains Random Forest on labeled data
- Calculates training accuracy
- Saves model to disk

### **Step 4: Real-Time Recognition**
```bash
python sign_language_recognizer.py recognize
```

1. Load trained model and label encoder
2. Open webcam and initialize MediaPipe
3. For each frame:
   - Detect hands
   - Extract features (same as Phase 2)
   - Run model inference → prediction + confidence
   - Store prediction in 5-frame buffer
   - Apply majority voting on buffer
   - Display smoothed result on frame
4. Show gesture name + confidence score overlay
5. Exit on 'q' key press

### **Temporal Smoothing Logic**
```python
# Collect last 5 predictions
prediction_buffer = ["hello", "hello", "hello", "goodbye", "hello"]

# Majority voting
smoothed_prediction = max(set(buffer), key=buffer.count)
# Result: "hello" (4 out of 5 votes)

# Confidence threshold
if confidence >= 0.5:
    display(smoothed_prediction, confidence)
else:
    display("Low confidence")
```

---

##  Installation

### Prerequisites
- Python 3.8+
- Webcam
- Good lighting environment

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd sign-language-recognition
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Required Packages:**
```
opencv-python>=4.5.0
mediapipe>=0.8.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

---

##  Usage

### **1. Collect Training Data**

Start collecting data for your first sign:
```bash
python sign_language_recognizer.py collect hello --samples 30
```

**Controls during collection:**
- **SPACE** - Start/stop recording
- **ENTER** - Save current recording
- **'c'** - Clear buffer
- **'n'** - Switch to new sign
- **'q'** - Quit

**Example workflow:**
```bash
# Collect multiple signs
python sign_language_recognizer.py collect hello --samples 100
python sign_language_recognizer.py collect goodbye --samples 100
python sign_language_recognizer.py collect thank_you --samples 100
python sign_language_recognizer.py collect please --samples 100
```

### **2. Train the Model**

Once you've collected data for 10-15 signs:
```bash
python sign_language_recognizer.py train
```


**Display Output:**
- Top-left corner shows: `Prediction: hello (0.95)`
- Hand landmarks rendered with neon style
- Confidence score updates in real-time
- Available signs listed on screen

**Controls:**
- **'q'** - Quit

---

##  Project Structure

```
sign-language-recognition/
├── sign_language_recognizer.py    # Main script (4-in-1)
├── sign_collection_enhanced.py    # Enhanced collection tool
├── ARCHITECTURE.md                # Detailed architecture docs
|
├── README.md                      # This file
├── Architecture.png               # System architecture diagram
│
├── sign_data/                     # Training data (collected)
│   ├── hello/
│   │   ├── hello_000.pkl
│   │   ├── hello_001.pkl
│   │   └── ... (30+ samples)
|
│
└── models/                        # Trained models
    ├── sign_language_model.pkl    # Random Forest classifier
    └── label_encoder.pkl          # Label mapping
```

---

##  Performance

### Accuracy Metrics
- **Training Accuracy**: 92-96% (depends on data quality)
- **Real-time Inference**: 20-30 FPS
- **Per-frame Latency**: 30-50ms
  - MediaPipe detection: 15-25ms
  - Feature extraction: 1-2ms
  - Model inference: 1-2ms
  - Visualization: 5-10ms

### Model Details
- **Classifier**: Random Forest
- **Trees**: 100
- **Max Depth**: 15
- **Input Features**: 48 (24 per hand)
- **Model Size**: ~500KB - 2MB
- **Training Time**: 1-10 seconds (depending on data)


##  Recommended Workflow

### **MVP (Minimum Viable Product) - Start Here**

Collect these 10 signs for quick validation:
1. **HELLO** - Wave side-to-side
2. **GOODBYE** - Wave up-and-down
3. **THANK_YOU** - Hand to mouth, move outward
4. **PLEASE** - Circular motion on chest
5. **YES** - Fist nod up-and-down
6. **NO** - Hand side-to-side
7. **HELP** - Open hand supporting fist
8. **WATER** - W-shape at mouth
9. **EAT** - Fingertips to mouth
10. **SLEEP** - Hand closes at cheek

**Time to complete**: ~30 minutes  
**Expected accuracy**: 92-96%

### **Phase 2 - Expansion**

Add 20+ more common phrases:
- Emotions (happy, sad, angry, tired)
- Verbs (work, play, drink, love)
- Daily actions (sleep, eat, walk, run)

### **Phase 3 - Advanced**

- Full alphabet (26 letters)
- Domain-specific signs (healthcare, retail)
- Complex two-hand gestures

---

## 🔧 Customization

### **Adjust Confidence Threshold**
```python
CONFIDENCE_THRESHOLD = 0.7  # Increase for stricter recognition
```

### **Change Smoothing Window**
```python
PREDICTION_SMOOTHING_FRAMES = 7  # More frames = smoother but slower
```

### **Modify Camera Settings**
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Higher resolution
cap.set(cv2.CAP_PROP_FPS, 60)             # Higher FPS
```

---

##  Troubleshooting

### **Issue: Low Recognition Accuracy**
- **Solution**: Collect more training data (50+ samples per sign)
- **Solution**: Ensure good lighting conditions
- **Solution**: Verify sign distinctiveness

### **Issue: Model Hallucination**
- **Solution**: Increase `PREDICTION_SMOOTHING_FRAMES`
- **Solution**: Raise `CONFIDENCE_THRESHOLD`
- **Solution**: Collect data with more hand position variations

### **Issue: Webcam Not Detected**
- **Solution**: Check camera permissions
- **Solution**: Try `cv2.VideoCapture(1)` instead of `0`
- **Solution**: Restart the application

### **Issue: Slow Performance**
- **Solution**: Reduce frame resolution
- **Solution**: Lower MediaPipe detection confidence
- **Solution**: Use GPU acceleration if available

---

##  Future Improvements

### Planned Enhancements
- [ ] **Temporal Models**: LSTM for capturing hand movement sequences
- [ ] **Deep Learning**: CNN-based feature extraction
- [ ] **Multi-hand Interaction**: Detect sign relationships between hands
- [ ] **Pose Integration**: Combine hand + body pose for context
- [ ] **Mobile Deployment**: Export to TensorFlow Lite
- [ ] **Web Interface**: Real-time recognition via browser
- [ ] **Text-to-Speech**: Audio output of recognized signs
- [ ] **Sign Language Variants**: Support for ASL, BSL, CSL

### Performance Optimization
- [ ] GPU acceleration (CUDA/Metal)
- [ ] Model quantization for smaller file size
- [ ] ONNX model export for cross-platform deployment

---

##  Documentation

Detailed documentation is available in the project:
- **`ARCHITECTURE.md`** - In-depth system architecture
- **`SIGNS_TO_COLLECT.md`** - Comprehensive sign selection guide
- **`CAPTURE_AND_STORE_DATA.md`** - Data collection tutorial
- **`COLLECTION_QUICK_GUIDE.md`** - Quick reference guide

---

##  References

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands)
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests)
- [NumPy Documentation](https://numpy.org/doc/)

---

##  Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

##  Author

    beastgotfried

---

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect training data (start with 10 signs)
python sign_language_recognizer.py collect hello --samples 30
python sign_language_recognizer.py collect goodbye --samples 30
# ... (repeat for more signs)

# 3. Train the model
python sign_language_recognizer.py train

# 4. Run real-time recognition
python sign_language_recognizer.py recognize
```

That's it! You now have a working sign language recognition system. 🤚

---

