# Quick Start Guide - SignBridge

## Running the Complete Pipeline

### Prerequisites
Make sure you have:
1. ✓ Virtual environment activated
2. ✓ All dependencies installed (opencv-python, mediapipe, scikit-learn, numpy)
3. ✓ Model trained and saved in `models/`
4. ✓ Webcam available and functional

### Method 1: Using main.py (Recommended)

Run the entire production pipeline with a single command:

```bash
# From project root
python main.py
```

This will:
1. Initialize the background service
2. Start the inference subprocess (webcam gesture recognition)
3. Start the bridge subprocess (gesture → text conversion)
4. Monitor process health and auto-restart on crashes
5. Provide real-time status updates
6. Handle graceful shutdown

### Method 2: Direct Service Run

If you prefer to run components directly:

```bash
# From src directory
cd src

# Run inference only
python phase4_inference.py

# Run bridge only (requires inference in background)
python prediction_bridge.py

# Run background service
python background_service.py
```

---

## Controls During Runtime

### In the Inference Window (Webcam)
- **Q** - Quit inference and shutdown
- **SPACE** - Pause/resume gesture recognition

### In Terminal
- **Ctrl+C** - Emergency shutdown (graceful cleanup)

---

## Understanding the Output

### Console Output
```
================================================================================
SIGNBRIDGE - Sign Language Detection Pipeline
================================================================================

Starting real-time gesture recognition system...

================================================================================
CONTROLS & INSTRUCTIONS
================================================================================

Inference Window (Webcam):
  • Press Q to quit inference
  • Press SPACE to pause/resume
  
[✓] SignBridge service started successfully!

Gesture Recognition Running - Wave your hands at the camera!
```

### Pipeline Status (printed every 15 seconds)
```
--------------------------------------------------------------------------------
PIPELINE STATUS
--------------------------------------------------------------------------------
  Running: True
  Uptime: 45.3s
  Inference Process: ✓ ALIVE
  Bridge Process: ✓ ALIVE
  Queue Size: 3 items
  Inference Restarts: 0
  Bridge Restarts: 0
--------------------------------------------------------------------------------
```

### Log Files
Logs are automatically saved to:
```
logs/signbridge_YYYYMMDD_HHMMSS.log
```

Each session creates a new log file with detailed debugging information.

---

## Troubleshooting

### Processes not starting
**Check logs:**
```bash
cat logs/signbridge_*.log
```

Look for:
- Model file not found
- Encoder file not found
- Webcam access denied
- Module import errors

**Solution:** Ensure model is trained and saved in `models/` directory

### Webcam not detected
**Error message:**
```
Error: cannot open webcam
```

**Solutions:**
- Ensure webcam is connected and not in use by another application
- Check device permissions
- Try different camera index in config.py (WEBCAM_INDEX)

### Queue overflow warnings
**Message:**
```
WARNING: Queue overflow - 150 items waiting
```

**Causes:**
- Inference producing faster than bridge consuming
- UI automation blocking on keystrokes
- Network/system lag

**Solution:**
- Check if active window is responsive
- Reduce inference FPS if needed

### Process crashes and auto-restart

The system will automatically restart crashed processes up to 3 times with exponential backoff:
- 1st crash → restart after 1 second
- 2nd crash → restart after 2 seconds  
- 3rd crash → restart after 4 seconds
- 4th crash → critical log and no more restarts

Check logs to debug the underlying issue.

---

## Performance Tuning

### Adjust gesture smoothing (reduce jitter)
Edit `src/config.py`:
```python
STABILITY_FRAMES = 7  # Increase from 5 for more smoothing
```

### Adjust confidence threshold (reduce false positives)
```python
CONFIDENCE_THRESHOLD = 0.80  # Increase from 0.75 for stricter filtering
```

### Adjust pause detection timing
```python
PAUSE_COMMIT_SECONDS = 3.0  # Increase for longer backspace hold
PAUSE_CONFIRM_SECONDS = 2.0  # Increase for longer pause before auto-commit
```

---

## Features Demo

### 1. Real-time Recognition
Wave individual hand gestures at the camera. Each distinct gesture generates a character or word.

### 2. Buffering
Gestures are buffered and displayed in the terminal. Type "hello" by gesturing H-E-L-L-O.

### 3. Word Completion
Hold a gesture for 2.5 seconds to trigger backspace (delete last character).

After 1.5 seconds with no hands detected, the current buffer commits as a word (adds space).

### 4. Keyboard Output (if pyautogui available)
Recognized text automatically types into the active window.

### 5. Feedback
- **TTS** (if pyttsx3 available): Speaks gesture label upon recognition
- **Notifications** (if win10toast available): Windows toast shows predictions

---

## Understanding the Pipeline

### Component Flow
```
┌─────────────────────────────────────────────────────────────┐
│ 1. WEBCAM INPUT                                             │
│    (30 FPS video capture)                                   │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. INFERENCE (phase4_inference.py - Subprocess)             │
│    • MediaPipe hand detection                               │
│    • Feature extraction (48-dim)                            │
│    • Random Forest prediction                               │
│    • Confidence filtering (≥75%)                            │
│    → Emits: {'type': 'prediction', 'label': 'A', ...}      │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. SHARED QUEUE (multiprocessing.Manager().Queue)           │
│    (Thread-safe, inter-process communication)              │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. BRIDGE (prediction_bridge.py - Subprocess)               │
│    • QueueConsumerThread (polls every 0.1s)                │
│    • Gesture → Text mapping                                 │
│    • Character buffering (BridgeState)                      │
│    • Token handling (backspace, commit)                     │
│    → Emits: {'action': 'type', 'content': 'a', ...}       │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. UI AUTOMATION (ui_automation.py)                         │
│    • pyautogui.write() (if available)                       │
│    • Fallback: Console logging                              │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. ACTIVE WINDOW                                            │
│    • Text appears in focused application                    │
│    • Works with any text input field                        │
└─────────────────────────────────────────────────────────────┘
```

### Background Monitoring
```
┌──────────────────────────────────────┐
│ HealthMonitor (Daemon Thread)        │
│ • Checks every 5 seconds             │
│ • Detects process crashes            │
│ • Auto-restart with backoff          │
│ • Monitors queue overflow/stalls     │
└──────────────────────────────────────┘
```

---

## Testing

### Run validation tests
```bash
cd src
python debug_test.py
```

Expected output:
```
✓ PASS  Imports                       
✓ PASS  Configuration                 
✓ PASS  UI Automation                 
✓ PASS  BridgeState                   
✓ PASS  PredictionBridge              
✓ PASS  BackgroundService             

Total: 6/6 tests passed
✓ ALL TESTS PASSED - Code is ready for integration!
```

---

## Next Steps

1. **Run main.py** to start the pipeline
2. **Wave gestures** at the camera
3. **Watch terminal** for real-time status
4. **Check logs** if any issues
5. **Adjust config** for your preferences
6. **Press Ctrl+C** to gracefully shutdown

---

## Additional Resources

- **PIPELINE_OVERVIEW.md** - Complete feature documentation
- **src/config.py** - Configuration parameters
- **logs/** - Session logs
- **models/** - Trained model files

Happy signing! 🤟
