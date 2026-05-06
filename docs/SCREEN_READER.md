# Screen Reader + UI Automation Integration for Sign-Language-Detection

## Plan
1. Add a background service that receives model predictions and types them into UI textboxes.
2. Implement a UI automation layer to enumerate and control text inputs (no GUI focus required when possible).
3. Add a prediction bridge (IPC) so the OpenCV/model process can safely send strings to the UI service.
4. Wire a small runner to test end-to-end (camera/model -> queue -> UI automation).
5. Add configuration, logging, rate-limiting (debounce), and tests.

## Dependencies
- Python 3.9+ (Windows recommended for UI automation)
- pip packages (add to `requirements.txt`):
	- `opencv-python` (>=4.5)
	- `mediapipe`
	- `numpy`
	- `scikit-learn` or `tensorflow`/`torch` (your model stack)
	- `uiautomation` (or `pywinauto[uia]`) — UI Automation for Windows
	- `pyautogui` (optional fallback keystrokes)
	- `pyzmq` (optional, if using ZeroMQ for IPC)
	- `flask` or `fastapi` (optional, if using HTTP IPC)
	- `python-decouple` or `pydantic` (optional config helpers)
- System notes:
	- Run the UI automation service with same or higher privilege as the target app (UAC can block).
	- If automating elevated apps, run the script elevated.

## High-level Features
- Background service (no UI focus required)
- Discover and list editable text inputs (textboxes) across windows
- Map predictions to target textbox(es) (configurable)
- Type/paste predicted text into controls using ValuePattern when available
- Fallback to focus + SendKeys for controls that lack ValuePattern
- Debounce/rate-limit predictions to avoid flooding inputs
- Support multiple IPC methods (in-process queue, TCP/HTTP, ZeroMQ)
- Logging, error handling, optional monitoring UI
- Security: whitelist/blacklist target apps; avoid injecting into elevated windows unintentionally

## Proposed New Files & Key Functions
- `src/config.py`
	- `load_config(path: str) -> dict` — load config (mappings, debounce, IPC)
- `src/prediction_bridge.py`
	- `PredictionQueue` — thread/process-safe queue wrapper
	- `start_tcp_server(host, port, on_message)` — simple TCP JSON receiver
	- `zmq_server(bind_addr, on_message)` — ZeroMQ receiver example
- `src/ui_automation.py`
	- `discover_textboxes() -> List[ControlInfo]`
	- `set_text(control_handle, text: str) -> bool`
	- `send_keys_fallback(control_handle, text: str) -> bool`
	- `map_controls(mapping_config) -> List[ControlInfo]`
- `src/background_service.py`
	- `UIService` with `start()`, `stop()`, `on_prediction(prediction: str)`
	- `debounce_and_apply(pred: str, target_controls: List[ControlInfo])`
- `src/model_runner.py`
	- `ModelRunner.predict_frame(frame) -> (label, confidence)`
	- `ModelRunner.start_camera_loop(output_queue)`
- `scripts/run_service.py`
	- CLI to launch service (with options: start model / listen only)
- `docs/SCREEN_READER.md` (this file)
- `tests/` for unit tests (use mocking for UI automation)

## Function-level Details
- discover_textboxes()
	- Walk desktop/window tree.
	- Return editable controls only (ControlType.Edit / className "Edit").
	- For each control, attempt to read ValuePattern to determine writability.
	- Return metadata: id/handle, name, window title, process id.
- set_text(control_handle, text)
	- Prefer direct write via UIA ValuePattern.SetValue.
	- Skip controls with `IsReadOnly`.
	- Support clipboard-paste: save clipboard, set clipboard, focus control, send Ctrl+V, restore clipboard.
- send_keys_fallback(control_handle, text)
	- Focus control, then use `uiautomation.SendKeys` or `pyautogui.typewrite`.
	- Make typing speed configurable to reduce missed characters.
- PredictionQueue / IPC
	- Thread-safe `put()` and `get()` with timeouts.
	- TCP JSON receiver example: parse JSON { "text": "...", "target": "..." } and call handler.
- UIService.on_prediction(pred)
	- Debounce repeated identical predictions for a configurable time window (e.g., 300 ms).
	- Optionally require a confidence threshold before typing.
	- Map prediction to targets: "all textboxes" or specific (via config).
	- Retry on transient failures with limited backoff.

## IPC Options (summary)
- In-process: `queue.Queue()` — simplest if model and service run in same process.
- Multiprocess: `multiprocessing.Queue()` — for local separate processes.
- TCP/HTTP: `socket` or `Flask/FastAPI` — easy cross-process integration.
- ZeroMQ: `pyzmq` — low-latency, good for high-throughput.
Recommendation: start with `multiprocessing.Queue` for local integration; add TCP/ZMQ if separating across machines/processes.

## Security & Robustness
- Whitelist/blacklist target windows and processes.
- Avoid sending to elevated windows from non-elevated processes.
- Log all injections; provide emergency kill switch.
- Rate-limit keystrokes; provide dry-run mode for testing.
- Provide configuration to restrict which apps are automated.

## Proposed File / Folder Structure
```
Sign-Language-Detection/
├─ README.md
├─ requirements.txt
├─ docs/
│  └─ SCREEN_READER.md
├─ data/
├─ models/
├─ notebooks/
├─ scripts/
│  └─ run_service.py
├─ src/
│  ├─ phase1_database.py
│  ├─ phase2_extract.py
│  ├─ phase3_train.py
│  ├─ phase4_inference.py
│  ├─ config.py
│  ├─ model_runner.py
│  ├─ prediction_bridge.py
│  ├─ ui_automation.py
│  ├─ background_service.py
│  └─ utils/
│     ├─ math_utils.py
│     ├─ ipc_utils.py
│     └─ typing_utils.py
├─ tests/
└─ logs/
```

## Minimal End-to-End Flow
1. `phase4_inference.py` / `model_runner` does inference and pushes `prediction_queue.put(pred_str)`.
2. `background_service.UIService` consumes predictions: `pred = queue.get()`.
3. `UIService` runs `discover_textboxes()` (or uses cached mapping) and calls `set_text()` for target controls.
4. Log success/failure; honor debounce and confidence thresholds.

## Next Steps I can implement for you
- Scaffold the new files with function skeletons and docstrings.
- Implement a working `PredictionQueue` + `UIService` example using `multiprocessing.Queue` and `uiautomation`.
- Add a TCP/ZMQ IPC example.

Tell me which of those to implement next, or say "create file" and I will move this into `docs/SCREEN_READER.md`.
