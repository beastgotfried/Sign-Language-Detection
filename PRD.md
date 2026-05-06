**SignBridge**

Sign Language to Text --- Accessibility Tool

**Product Requirements Document**

Version 1.0

May 2026

+-----------------------------------+-----------------------------------+
| **Target Users**                  | **Primary Platform**              |
|                                   |                                   |
| Blind / Visually Impaired         | Windows Desktop                   |
+-----------------------------------+-----------------------------------+

**1. Executive Summary**

SignBridge is a real-time accessibility tool that translates hand gestures captured through a webcam into typed text, delivered directly into any focused chatbox or text input on a user\'s screen. It is designed from the ground up for blind and visually impaired users, requiring zero keyboard interaction and providing full audio feedback through a text-to-speech engine at every step.

The system is trained offline on an image-based sign language dataset, allowing it to recognise a defined vocabulary of hand gestures. During live use, a webcam captures the user\'s hands continuously, the trained model infers the intended word from each stable gesture, speaks that word aloud so the user can verify it, and then types it immediately into the active text input. A dedicated backspace gesture allows word-level correction without any keyboard involvement.

SignBridge addresses a gap in existing assistive technology: most voice-to-text tools require audible speech, which is not always possible or desirable, while most sign language tools are demonstration systems that display recognised gestures on screen but do not integrate with real applications. SignBridge closes that gap by making the output directly actionable inside any application the user is already running.

  ---------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Goal**   Enable a blind sign language user to compose and send messages in any chat application without touching a keyboard, using only hand gestures and audio feedback.

  ---------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------

**2. Problem Statement**

**2.1 User Need**

Blind and visually impaired individuals who communicate using sign language face a significant barrier when using digital devices: standard keyboards require sighted navigation, voice-to-text requires audible speech in environments where that may not be appropriate, and no existing tool translates sign gestures directly into application input with audio confirmation.

**2.2 Existing Gap**

  -------------------------------------------------------------------------------------------------------------------------
  **Existing Tool**                         **Limitation for Target Users**
  ----------------------------------------- -------------------------------------------------------------------------------
  Voice-to-text (e.g. Whisper, Dictation)   Requires audible speech; unusable in quiet or public environments

  Sign language recognition demos           Output displayed on screen only; no integration with real applications

  Screen reader + keyboard                  Requires sighted navigation to find and activate text inputs

  Eye tracking tools                        Expensive, requires precise calibration, not usable by all visual impairments

  Switch access / AAC devices               Slow, limited vocabulary, not integrated with mainstream chat apps
  -------------------------------------------------------------------------------------------------------------------------

**2.3 Success Criteria**

-   User can compose and send a multi-word message in any chat application without touching a keyboard

-   Every recognised word is spoken aloud before being typed

-   User can correct the last word using a single backspace gesture

-   System achieves greater than 90% word-level accuracy on the supported vocabulary under normal indoor lighting

-   End-to-end latency from gesture completion to word spoken is under 500ms

**3. System Architecture**

SignBridge operates as two loosely coupled processes communicating through a shared queue. This separation ensures that the computationally heavy inference loop does not block the UI automation layer, and vice versa.

**3.1 High-Level Pipeline**

  ------------- --------------------------------------------------------------------------------------------------------------
  **Phase 1**   Image Dataset → MediaPipe feature extraction → Random Forest training → saved model file (offline, run once)

  ------------- --------------------------------------------------------------------------------------------------------------

  ------------- -------------------------------------------------------------------------------------------------
  **Phase 2**   Live webcam → OpenCV frame capture → MediaPipe landmark extraction → feature vector computation

  ------------- -------------------------------------------------------------------------------------------------

  ------------- ------------------------------------------------------------------------------------------------------------
  **Phase 3**   Feature vector → RF inference → confidence scoring → stability buffer (5-frame confidence-weighted voting)

  ------------- ------------------------------------------------------------------------------------------------------------

  ------------- -----------------------------------------------------------------------------------------------------
  **Phase 4**   Stable high-confidence prediction → word buffer → TTS speaks word → word typed into focused chatbox

  ------------- -----------------------------------------------------------------------------------------------------

  ------------- --------------------------------------------------------------------------------------------------------------------
  **Phase 5**   Backspace gesture detected → last word removed from buffer → last word deleted from chatbox → TTS confirms removal

  ------------- --------------------------------------------------------------------------------------------------------------------

**3.2 Process Architecture**

The system runs as two OS-level processes connected by a multiprocessing.Queue:

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Process**            **Responsibility**                                                                                                **Key Libraries**
  ---------------------- ----------------------------------------------------------------------------------------------------------------- --------------------------------------------------------
  Inference Process      Webcam capture, MediaPipe landmark extraction, RF model inference, stability buffering, confidence thresholding   OpenCV, MediaPipe, scikit-learn, NumPy

  UI Service Process     Consuming prediction queue, TTS speech, clipboard-paste to chatbox, word buffer management, backspace handling    pyttsx3 / edge-tts, pywinauto, pyperclip, uiautomation
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**3.3 Data Flow Detail**

Within the Inference Process, each webcam frame goes through the following steps in sequence:

1.  OpenCV captures frame at target FPS (20-30)

2.  MediaPipe Hands detects up to 2 hands and extracts 21 landmarks (x, y, z) per hand

3.  Feature engineering layer computes 48 geometric features: 8 distance features, 6 angle features, 5 curl/flexion features, 3 position features per hand

4.  Random Forest classifier outputs a class label and a probability distribution across all classes

5.  Confidence-weighted stability buffer accumulates last 5 frames, weighing each frame by its confidence score rather than treating all frames equally

6.  If the dominant prediction holds for 5 frames and the weighted confidence exceeds the threshold (default 0.75), the prediction is committed and pushed to the queue

Within the UI Service Process:

1.  Word is dequeued from the multiprocessing.Queue

2.  If word is the backspace signal, last word is removed from buffer, deleted from chatbox, and TTS speaks \'removed: \[word\]\'

3.  Otherwise, word is appended to the sentence buffer

4.  TTS immediately speaks the word aloud (interrupting any current speech)

5.  Word is typed into the focused chatbox via clipboard-paste

6.  If no gesture is detected for a configurable pause window (default 2.5 seconds) and the buffer is non-empty, TTS reads back the full sentence, then auto-commits

**4. Feature Specification**

**4.1 Training Pipeline (Offline)**

**F-01: Dataset-based feature extraction**

The system shall accept a directory of labelled hand gesture images as training input. For each image, MediaPipe shall attempt landmark extraction. Images where MediaPipe fails to detect a hand shall be logged and skipped rather than crashing the pipeline. The extracted 48-dimensional feature vectors and their labels shall be serialised to disk for use in training.

**F-02: Random Forest training**

The training script shall train a Random Forest classifier with 100 decision trees, max depth 15, and minimum 5 samples per leaf. The script shall produce a train/test split (80/20) and report both training and held-out test accuracy. The trained model and label encoder shall be saved as pickle files. Training shall complete in under 60 seconds for a dataset of up to 10,000 images.

**F-03: Validation reporting**

After training, the script shall print a per-class accuracy table showing which gestures perform below an acceptable threshold (default 85%). Gestures falling below threshold shall be flagged as requiring additional training samples. A confusion matrix shall be saved to disk for post-training analysis.

**4.2 Inference Engine (Live)**

**F-04: Continuous webcam capture**

The inference process shall open the default webcam on launch and maintain continuous frame capture. Frame rate shall be configurable. The process shall handle webcam disconnection gracefully, logging the event and attempting reconnection rather than crashing.

**F-05: Confidence-weighted stability buffer**

Rather than simple majority voting, each frame\'s prediction shall be weighted by its confidence score. A prediction is committed only when the weighted average across the last 5 frames exceeds the configured threshold (default 0.75). This means a single very high-confidence frame can outweigh several low-confidence frames of a different class, producing faster and more accurate commitment than equal-weight voting.

**F-06: Backspace gesture as first-class label**

The backspace gesture shall be a fully trained class in the Random Forest, not a post-processing rule. This means it benefits from the same confidence thresholding and stability buffering as all other gestures. The gesture chosen for backspace shall be maximally distinct from all ASL/vocabulary gestures --- a two-handed symmetrical motion is recommended. The backspace class shall be included in the training dataset with the same sample count as other classes.

**F-07: No-hand and low-confidence silence**

When no hand is detected, or when the stability buffer confidence is below threshold, the system shall produce no output --- no speech and no typing. Silence is always preferable to a wrong word being spoken and typed. The TTS shall not speak \'unclear\' or similar filler phrases as this creates noise for the user.

**4.3 Audio Feedback (TTS)**

**F-08: Immediate word speech**

Every committed word shall be spoken aloud within 300ms of being committed. The TTS engine shall interrupt any currently playing audio before speaking the new word. Queuing speech is explicitly prohibited --- a backlog of unspoken words would disorient the user.

**F-09: Audio event vocabulary**

  ----------------------------------------------------------------------------------------------
  **Event**                               **TTS Output**
  --------------------------------------- ------------------------------------------------------
  Word committed                          Speaks the word naturally (e.g. \'hello\')

  Backspace gesture                       Says \'removed: \[word\]\' (e.g. \'removed: hello\')

  Sentence auto-committed (paste fired)   Reads back full sentence, then says \'sent\'

  Pause window reached, buffer empty      Silence --- no output

  Chatbox not found / paste failed        Says \'no target\'

  Model loaded and ready                  Says \'ready\'

  Low confidence sustained (5+ seconds)   Says \'unclear\' once, then silent
  ----------------------------------------------------------------------------------------------

**F-10: TTS engine selection**

The system shall support two TTS backends selectable via configuration: pyttsx3 for fully offline operation (lower voice quality), and edge-tts for online operation (significantly better voice quality, requires internet). The default shall be pyttsx3 to ensure the tool works without connectivity. Speech rate and volume shall be configurable.

**4.4 UI Automation Layer**

**F-11: Focus-based target locking**

The system shall not attempt to discover or enumerate all open chatboxes. Instead, it shall monitor the currently focused window and control. When the user focuses a text input (by clicking into it with a mouse or by other means), the UI service shall lock onto that control as the typing target. This sidesteps the unreliable auto-discovery problem and works across native apps, Electron apps, and browsers.

**F-12: Tiered typing strategy**

The UI service shall attempt the following strategies in order, falling back to the next if the previous fails:

7.  Clipboard-paste: set system clipboard to the word, ensure target is focused, send Ctrl+V, restore original clipboard content. This is the primary strategy and works for virtually all application types including browsers and Electron apps.

8.  UIA ValuePattern: use Windows UI Automation ValuePattern.SetValue for native Windows controls that support it. Faster than clipboard-paste but only available for standard Win32/WPF controls.

9.  Simulated keystrokes: use pyautogui.typewrite as last resort. Handles ASCII reliably but may drop characters at high typing speeds; typing speed shall be configurable.

**F-13: Word-level backspace**

When the backspace gesture is received, the UI service shall delete the last committed word from the chatbox. Since the word was inserted via clipboard-paste, deletion requires sending a calculated number of Backspace keystrokes equal to the length of the last word plus one (for the trailing space). The word shall also be removed from the in-memory sentence buffer. The TTS shall confirm with \'removed: \[word\]\'.

**F-14: Pause-based sentence commit**

If no new word has been added to the buffer for a configurable duration (default 2.5 seconds) and the buffer contains at least one word, the system shall enter a commit window: TTS reads back the full sentence, then waits 1.5 seconds. If a backspace gesture arrives during that window, it applies normally. If no gesture arrives, the sentence is considered final --- no additional action is taken since each word was already typed as it was committed. The buffer is cleared and the system returns to listening.

**5. File and Folder Structure**

The following is the complete target structure for the project. Existing files from the original codebase are noted; new files to be created are marked as NEW.

  ------------------------------------------------------------------------------------------------------------------------------------------------------
  **Path**                         **Purpose**                                                                                              **Status**
  -------------------------------- -------------------------------------------------------------------------------------------------------- ------------
  requirements.txt                 All Python dependencies including new additions                                                          UPDATE

  README.md                        Project overview and quick-start guide                                                                   UPDATE

  ARCHITECTURE.md                  Deep technical architecture reference                                                                    UPDATE

  src/phase1_dataset.py            Offline dataset loading and MediaPipe feature extraction for training                                    NEW

  src/phase2_features.py           Feature engineering functions (distances, angles, curl, position) --- shared by training and inference   REFACTOR

  src/phase3_train.py              Random Forest training, validation reporting, confusion matrix, model serialisation                      REFACTOR

  src/phase4_inference.py          Live webcam loop, stability buffer, confidence-weighted voting, queue push                               REFACTOR

  src/config.py                    Central configuration dataclass --- all tunable parameters in one place                                  NEW

  src/model_runner.py              ModelRunner class wrapping phase4 inference for clean process separation                                 NEW

  src/prediction_bridge.py         multiprocessing.Queue wrapper, signal definitions (BACKSPACE token, COMMIT token)                        NEW

  src/tts_service.py               TTS engine abstraction, interrupt support, audio event vocabulary                                        NEW

  src/ui_automation.py             Focus-based target locking, tiered paste strategy, word-level backspace                                  NEW

  src/background_service.py        UIService class --- consumes queue, manages word buffer, coordinates TTS and UI automation               NEW

  src/word_buffer.py               WordBuffer class --- ordered word list, undo tracking, character-count tracking for backspace            NEW

  scripts/run_service.py           CLI entry point --- launches inference process and UI service process                                    NEW

  scripts/collect_dataset.py       Utility to validate and inspect a training dataset before running training                               NEW

  tests/test_features.py           Unit tests for feature engineering functions                                                             NEW

  tests/test_buffer.py             Unit tests for WordBuffer --- add, undo, clear, commit                                                   NEW

  tests/test_tts.py                Mock-based tests for TTS audio event vocabulary                                                          NEW

  tests/test_ui_automation.py      Mock-based tests for paste strategies and backspace calculation                                          NEW

  models/sign_language_model.pkl   Trained Random Forest classifier (generated by training)                                                 GENERATED

  models/label_encoder.pkl         Label encoder mapping gesture names to integers                                                          GENERATED

  data/                            Training dataset root --- one subfolder per gesture class                                                EXISTING

  logs/                            Runtime logs from inference and UI service processes                                                     NEW

  docs/SCREEN_READER.md            UI automation and IPC technical reference                                                                UPDATE
  ------------------------------------------------------------------------------------------------------------------------------------------------------

**6. Module Specifications**

**6.1 src/config.py**

Central configuration file. All tunable values live here and nowhere else. Other modules import from config rather than hardcoding values.

  ---------------------------------------------------------------------------------------------------------
  **Parameter**           **Default / Description**
  ----------------------- ---------------------------------------------------------------------------------
  CONFIDENCE_THRESHOLD    0.75 --- minimum weighted confidence to commit a prediction

  STABILITY_FRAMES        5 --- number of frames in the stability buffer

  PAUSE_COMMIT_SECONDS    2.5 --- seconds of silence before sentence readback begins

  PAUSE_CONFIRM_SECONDS   1.5 --- seconds after readback before buffer is cleared

  TTS_BACKEND             \'pyttsx3\' or \'edge-tts\'

  TTS_RATE                175 --- words per minute for pyttsx3

  WEBCAM_INDEX            0 --- OpenCV VideoCapture index

  TARGET_FPS              25 --- target inference frame rate

  BACKSPACE_TOKEN         \'\_\_BACKSPACE\_\_\' --- sentinel string pushed to queue for backspace gesture

  TYPING_STRATEGY         \'clipboard\' --- primary typing method

  LOG_LEVEL               \'INFO\'
  ---------------------------------------------------------------------------------------------------------

**6.2 src/phase2_features.py**

Shared feature engineering module used by both the offline training pipeline and the live inference loop. Having a single source for feature computation ensures training and inference operate on identical feature representations --- a mismatch here is a common and hard-to-debug source of accuracy degradation.

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Function**                                  **Description**
  --------------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------
  extract_features(landmarks) -\> np.ndarray    Takes raw MediaPipe landmark list for one or two hands, returns 48-dimensional feature vector. Returns zero vector if no hand detected.

  compute_distances(landmarks) -\> np.ndarray   8 Euclidean distances: each fingertip to wrist, normalised by hand bounding box size to be scale-invariant

  compute_angles(landmarks) -\> np.ndarray      6 inter-finger angles using dot product of finger direction vectors

  compute_curl(landmarks) -\> np.ndarray        5 curl scores per finger: ratio of fingertip-to-palm distance vs finger length, approximating flexion

  compute_position(landmarks) -\> np.ndarray    3 values: wrist x, wrist y, wrist z normalised to frame dimensions

  normalise_landmarks(landmarks) -\> list       Translates landmarks so wrist is at origin, scales by hand bounding box --- makes features invariant to hand position and size in frame
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**6.3 src/prediction_bridge.py**

Thin wrapper around multiprocessing.Queue that defines the contract between the inference process and the UI service process.

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Component**         **Description**
  --------------------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  PredictionQueue       Class wrapping multiprocessing.Queue with put(word: str) and get(timeout) methods. Passes plain strings. BACKSPACE_TOKEN is the agreed sentinel for backspace events.

  BACKSPACE_TOKEN       Module-level constant \'\_\_BACKSPACE\_\_\' --- the inference process pushes this when the backspace gesture is detected, the UI service checks for it on dequeue
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**6.4 src/tts_service.py**

TTS abstraction layer. The rest of the system calls speak(text) and never interacts with a specific engine directly. This makes swapping TTS backends a single config change.

  ---------------------------------------------------------------------------------------------------------------------------------------------
  **Method**                      **Description**
  ------------------------------- -------------------------------------------------------------------------------------------------------------
  speak(text: str)                Interrupts any current speech and speaks text immediately. Non-blocking --- runs speech in a daemon thread.

  speak_word(word: str)           Speaks a committed word --- used for normal gesture recognition

  speak_removed(word: str)        Speaks \'removed: {word}\' --- used after backspace gesture

  speak_sentence(sentence: str)   Reads back the full sentence buffer --- used before auto-commit

  speak_event(event: str)         Speaks system events: \'ready\', \'sent\', \'no target\'

  stop()                          Stops any current speech immediately

  \_interrupt()                   Internal: cancels pyttsx3 runLoop or kills edge-tts subprocess
  ---------------------------------------------------------------------------------------------------------------------------------------------

**6.5 src/word_buffer.py**

The sentence buffer tracks not just words but also the character count of each word as typed, enabling precise word-level backspace without having to re-examine the chatbox state.

  ---------------------------------------------------------------------------------------------------------------------------
  **Method**                   **Description**
  ---------------------------- ----------------------------------------------------------------------------------------------
  add(word: str)               Appends word to buffer, records len(word)+1 (word + trailing space) as typed character count

  undo() -\> str \| None       Removes and returns last word. Returns None if buffer is empty.

  last_typed_chars() -\> int   Returns character count of last word as typed, used to calculate backspace keystrokes

  to_sentence() -\> str        Returns all buffered words joined by spaces

  clear()                      Empties the buffer after sentence commit

  is_empty() -\> bool          True if no words are buffered
  ---------------------------------------------------------------------------------------------------------------------------

**6.6 src/ui_automation.py**

Handles all interaction with the target application. Stateless --- receives word and acts, does not track buffer state (that is word_buffer\'s job).

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Function**                                    **Description**
  ----------------------------------------------- -----------------------------------------------------------------------------------------------------------------------------
  paste_word(word: str) -\> bool                  Attempts tiered paste: clipboard-paste first, UIA ValuePattern second, simulated keystrokes third. Returns True on success.

  backspace_chars(n: int) -\> bool                Sends n Backspace keystrokes to the focused control. Used for word-level undo.

  get_focused_control() -\> ControlInfo \| None   Returns metadata about the currently focused UI control. Returns None if focused control is not a text input.

  is_text_input(control) -\> bool                 Returns True if the control is editable: ControlType.Edit, or a content-editable web element.

  \_clipboard_paste(word: str) -\> bool           Saves clipboard, sets clipboard to word, sends Ctrl+V, restores clipboard. Works for browsers and Electron apps.

  \_uia_set_value(control, word: str) -\> bool    Uses Windows UIA ValuePattern.SetValue. Faster than clipboard but only for native Win32/WPF controls.

  \_keystroke_type(word: str) -\> bool            Uses pyautogui.typewrite at configured typing speed. Last resort fallback.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**6.7 src/background_service.py**

The main orchestrator of the UI service process. Coordinates word_buffer, tts_service, and ui_automation in response to predictions from the queue.

  --------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Method**                 **Description**
  -------------------------- -----------------------------------------------------------------------------------------------------------------------------------
  start(queue)               Starts the service loop consuming from queue. Blocks until stop() is called.

  stop()                     Signals the loop to exit cleanly.

  on_prediction(word: str)   Core handler. Checks for BACKSPACE_TOKEN, otherwise adds to buffer, speaks, and pastes. Resets the pause timer.

  \_handle_backspace()       Calls buffer.undo(), calculates backspace chars, calls ui_automation.backspace_chars(), calls tts.speak_removed()

  \_check_pause_timeout()    Run on a timer thread. If pause window exceeded and buffer non-empty, triggers sentence readback and post-readback commit window.

  \_commit_sentence()        Reads back sentence via TTS, waits for confirm window, clears buffer, logs commit.
  --------------------------------------------------------------------------------------------------------------------------------------------------------------

**7. Build Workflow**

The following is the recommended implementation sequence. Each step produces a testable deliverable before moving to the next. Do not skip steps --- each layer depends on the one below it being stable.

**Step 1 --- Environment and Configuration**

-   Create virtual environment and install dependencies

-   Write src/config.py with all default values

-   Verify MediaPipe, OpenCV, scikit-learn, pyttsx3 all import cleanly

-   Testable outcome: python -c \'from src.config import \*\' runs without error

**Step 2 --- Feature Engineering Module**

-   Write src/phase2_features.py with all feature functions

-   Write tests/test_features.py --- unit test each function with synthetic landmark data

-   Verify normalise_landmarks makes features scale and position invariant

-   Testable outcome: all unit tests pass; feature vector is consistently 48 dimensions

**Step 3 --- Training Pipeline**

-   Write src/phase1_dataset.py --- dataset loading and per-image MediaPipe extraction

-   Write src/phase3_train.py --- RF training with train/test split, per-class accuracy report, confusion matrix

-   Run on your dataset; verify test accuracy exceeds 90% on all classes

-   Flag any class below 85% and collect additional samples before proceeding

-   Testable outcome: models/sign_language_model.pkl and models/label_encoder.pkl produced; test accuracy report printed

**Step 4 --- Inference Loop**

-   Write src/phase4_inference.py with confidence-weighted stability buffer

-   Write src/model_runner.py wrapping phase4 in a clean class interface

-   Run in isolation with webcam --- print committed predictions to console

-   Verify backspace gesture is reliably detected and produces BACKSPACE_TOKEN

-   Verify no-hand and low-confidence frames produce no output

-   Testable outcome: console shows stable word predictions matching gestures, with no spurious outputs

**Step 5 --- Prediction Bridge**

-   Write src/prediction_bridge.py

-   Write a minimal two-process test script: producer pushes test strings, consumer prints them

-   Verify queue does not block either process and handles rapid bursts without dropping messages

-   Testable outcome: producer and consumer communicate correctly across processes

**Step 6 --- Word Buffer**

-   Write src/word_buffer.py

-   Write tests/test_buffer.py --- test add, undo, character counting, to_sentence, clear

-   Pay particular attention to undo on empty buffer (should return None without crashing)

-   Testable outcome: all unit tests pass

**Step 7 --- TTS Service**

-   Write src/tts_service.py with pyttsx3 backend first

-   Write tests/test_tts.py using mocks --- verify correct method is called for each event type

-   Test interrupt behaviour manually: trigger two words in rapid succession, verify second one cuts off first

-   Add edge-tts backend behind config flag once pyttsx3 is stable

-   Testable outcome: all audio events produce correct speech; interruption works correctly

**Step 8 --- UI Automation Layer**

-   Write src/ui_automation.py starting with clipboard-paste only

-   Write tests/test_ui_automation.py with mocked clipboard and focus APIs

-   Test clipboard-paste manually in Notepad, then a browser textarea, then Discord or Slack

-   Add backspace_chars and verify word-level deletion works correctly

-   Add UIA ValuePattern as second-tier strategy

-   Testable outcome: words appear correctly in target applications; backspace removes exactly the last word

**Step 9 --- Background Service**

-   Write src/background_service.py --- wire together word_buffer, tts_service, and ui_automation

-   Test with mocked queue: push a sequence of words and one BACKSPACE_TOKEN, verify buffer state and TTS calls are correct

-   Test pause timer: push words, wait for pause window, verify sentence readback fires

-   Testable outcome: full orchestration works correctly with mocked inputs

**Step 10 --- Integration and CLI**

-   Write scripts/run_service.py --- launch inference process and UI service process

-   Run end-to-end: sign gestures into webcam, verify words appear in a test chatbox with correct audio feedback

-   Test backspace gesture end-to-end: type two words, sign backspace, verify second word removed from chatbox and TTS says \'removed: \[word\]\'

-   Test pause commit: type a sentence, wait, verify TTS reads it back and buffer clears

-   Testable outcome: full pipeline works without keyboard interaction

**Step 11 --- Hardening and Edge Cases**

-   Test with different applications: Notepad, Chrome (Gmail), Discord, WhatsApp Web

-   Test with elevated privilege applications --- document any limitations found

-   Test rapid signing: verify system does not queue up speech or duplicate words

-   Test webcam disconnection: verify graceful handling and reconnection attempt

-   Run for 15 minutes continuously: verify no memory leaks or process hangs

-   Testable outcome: no crashes, no hung processes, no duplicate words in any tested application

**8. Dependencies**

The following packages shall be added to requirements.txt. Versions shown are minimum compatible versions.

  --------------------------------------------------------------------------------------------------
  **Package**        **Version**   **Purpose**
  ------------------ ------------- -----------------------------------------------------------------
  opencv-python      \>=4.5.0      Webcam capture and frame processing

  mediapipe          \>=0.10.0     Hand landmark detection and extraction

  numpy              \>=1.21.0     Feature vector computation

  scikit-learn       \>=1.0.0      Random Forest training and inference

  pyttsx3            \>=2.90       Offline TTS engine (primary)

  edge-tts           \>=6.1.0      Online neural TTS engine (optional, higher quality)

  pywinauto          \>=0.6.8      Windows UI Automation and control interaction

  uiautomation       \>=2.0.18     Windows UIA tree walking and ValuePattern access

  pyperclip          \>=1.8.2      Cross-process clipboard read/write for clipboard-paste strategy

  pyautogui          \>=0.9.54     Keystroke simulation fallback

  psutil             \>=5.9.0      Process monitoring and privilege detection
  --------------------------------------------------------------------------------------------------

  ---------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Note**   mediapipe\>=0.10.0 has API changes from 0.8.x. If the original codebase used 0.8.x, the landmark extraction calls in phase2_features.py must use the updated API (mp.solutions.hands vs mp.tasks.vision.HandLandmarker). Verify this before starting Step 2.

  ---------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**9. Risks and Mitigations**

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Risk**                                                            **Impact**                                                **Mitigation**
  ------------------------------------------------------------------- --------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------
  Training dataset lighting conditions differ from user environment   High --- reduced real-world accuracy                      Apply augmentation during training: brightness jitter, blur, horizontal flip. Test model under varied lighting before deployment.

  mediapipe fails on certain hand orientations or skin tones          Medium --- some gestures not detected                     Track MediaPipe failure rate per gesture class during validation. Add hard cases to training set. Silence is the correct fallback --- never guess.

  Backspace gesture accidentally fires during normal signing          High --- corrupts composed text with no visual feedback   Use two-handed symmetrical gesture not in ASL vocabulary. Set higher confidence threshold for backspace class (0.85+) than other classes.

  Clipboard-paste overwrites user\'s clipboard content                Low-Medium --- user loses clipboard                       Save and restore clipboard contents around every paste operation. Restore happens in a finally block to guarantee execution.

  UI automation blocked by UAC on elevated apps                       Medium --- tool fails silently in some apps               Detect elevated target via psutil. Speak \'no target\' rather than hanging. Document known limitations clearly.

  TTS speech lags behind signing pace                                 High --- audio feedback meaningless if always late        Use interrupt-on-new-word approach. Test with fastest realistic signing speed. Prefer pyttsx3 for local latency even if quality is lower.

  Electron and browser apps reject focus-based paste                  Medium --- tool unusable in some popular chat apps        Test clipboard-paste specifically in target apps early. Electron generally accepts Ctrl+V; browser textareas require focus to be confirmed.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**10. Out of Scope (Version 1.0)**

The following are explicitly deferred to future versions to keep v1.0 focused and deliverable:

-   macOS and Linux support --- UI automation layer is Windows-specific in v1.0

-   Mobile platform support

-   Full ASL alphabet (fingerspelling) --- v1.0 supports word-level gestures only

-   Two-hand interaction gestures beyond backspace

-   Custom vocabulary builder UI

-   Cloud model hosting or over-the-air model updates

-   Integration with screen readers (NVDA, JAWS) via MSAA --- valuable but complex

-   LSTM or CNN-based models --- Random Forest is the v1.0 baseline

-   Multi-user profiles with separate gesture vocabularies