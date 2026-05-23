"""
Prediction Bridge

Consumes predictions from phase4_inference.py queue and converts them to text output.

Parts:
1. Queue Consumer Thread - Non-blocking listener (DONE)
2. Bridge State Management - Buffer and history tracking (DONE)
3. Gesture Mapping & Prediction Processing - Map gestures to text
4. Token Processing - Handle BACKSPACE and COMMIT tokens
5. Output Handler - Send commands to UI automation
6. Feedback Systems - TTS and notifications (optional)
"""

import logging
import threading
import time
from queue import Queue, Empty
from typing import Dict, Any, Optional, List

# Import config constants
try:
    from config import (
        CONFIDENCE_THRESHOLD,
        BACKSPACE_TOKEN,
        COMMIT_TOKEN,
        GESTURE_TO_TEXT,
        QUEUE_TIMEOUT,
    )
except ImportError as e:
    # Will be set to defaults below
    pass

# Import optional dependencies
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from win10toast import ToastNotifier
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False

# Import UI automation
try:
    from ui_automation import send_to_ui
    UI_AUTOMATION_AVAILABLE = True
except ImportError:
    UI_AUTOMATION_AVAILABLE = False

# Disable logging output (no file or console logging)
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

# Set config defaults (fallback if import fails)
CONFIDENCE_THRESHOLD = 0.75
BACKSPACE_TOKEN = "__BACKSPACE__"
COMMIT_TOKEN = "__COMMIT__"
GESTURE_TO_TEXT = {
    'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e',
    'F': 'f', 'G': 'g', 'H': 'h', 'I': 'i', 'J': 'j',
    'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'O': 'o',
    'P': 'p', 'Q': 'q', 'R': 'r', 'S': 's', 'T': 't',
    'U': 'u', 'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y', 'Z': 'z',
}
QUEUE_TIMEOUT = 0.1

# Log availability of optional features
if not TTS_AVAILABLE:
    logger.debug("pyttsx3 not available - TTS disabled")
if not TOAST_AVAILABLE:
    logger.debug("win10toast not available - notifications disabled")
if not UI_AUTOMATION_AVAILABLE:
    logger.debug("ui_automation not available - output to file/log only")


class BridgeState:
    """
    State management for prediction bridge.
    
    Tracks:
    - Current word being typed (buffer)
    - All emitted text (history)
    - Last gesture for deduplication
    - Timing for rate limiting
    """
    
    def __init__(self):
        """Initialize bridge state with empty buffers"""
        self.current_buffer: str = ""
        self.output_history: List[str] = []
        self.last_label: Optional[str] = None
        self.last_emission_time: float = 0.0
        self.pending_space: bool = False
        
        logger.debug("BridgeState initialized")
    
    def add_char(self, char: str) -> None:
        """Add character to current buffer"""
        self.current_buffer += char
        logger.debug(f"Buffer updated: '{self.current_buffer}'")
    
    def emit_word(self, add_space: bool = True) -> str:
        """Emit current buffer as a word and clear buffer"""
        if not self.current_buffer:
            logger.debug("Buffer empty, nothing to emit")
            return ""
        
        word = self.current_buffer
        if add_space:
            word += " "
        
        self.output_history.append(word)
        self.current_buffer = ""
        
        logger.info(f"Word emitted: '{word.strip()}' | History: {len(self.output_history)} words")
        return word
    
    def backspace(self, count: int = 1) -> None:
        """Delete N characters from current buffer"""
        if not self.current_buffer:
            logger.debug("Buffer empty, cannot backspace")
            return
        
        deleted = self.current_buffer[-count:] if count <= len(self.current_buffer) else self.current_buffer
        self.current_buffer = self.current_buffer[:-count]
        
        logger.debug(f"Backspace {count} chars: deleted='{deleted}' | Buffer: '{self.current_buffer}'")
    
    def get_history(self) -> List[str]:
        """Get entire output history"""
        return self.output_history.copy()
    
    def get_full_output(self) -> str:
        """Get complete output as single string"""
        full = "".join(self.output_history) + self.current_buffer
        return full
    
    def reset(self) -> None:
        """Clear all state"""
        self.current_buffer = ""
        self.output_history = []
        self.last_label = None
        self.last_emission_time = 0.0
        self.pending_space = False
        
        logger.info("BridgeState reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get diagnostic statistics"""
        return {
            "buffer_length": len(self.current_buffer),
            "buffer_content": self.current_buffer,
            "history_length": len(self.output_history),
            "total_output": self.get_full_output(),
            "last_label": self.last_label,
            "last_emission_time": self.last_emission_time,
        }


class PredictionBridge:
    """
    Main bridge class that connects inference to UI automation.
    
    Responsibilities:
    - Consume predictions from queue
    - Map gestures to text
    - Manage state and buffering
    - Process tokens (backspace, commit)
    - Send output to UI layer
    - Provide feedback (TTS, notifications)
    """
    
    def __init__(self, input_queue: Queue, gesture_map: Optional[Dict[str, str]] = None):
        """
        Initialize prediction bridge.
        
        Args:
            input_queue: Queue from phase4_inference
            gesture_map: Custom gesture to text mapping (uses GESTURE_TO_TEXT if not provided)
        """
        self.input_queue = input_queue
        self.gesture_map = gesture_map or GESTURE_TO_TEXT
        self.state = BridgeState()
        self.consumer = QueueConsumerThread(input_queue, queue_timeout=QUEUE_TIMEOUT)
        
        # Register handlers with consumer
        self.consumer.on_prediction = self._process_prediction
        self.consumer.on_token = self._process_token
        
        logger.info("PredictionBridge initialized")
    
    def start(self):
        """Start consuming events from queue"""
        self.consumer.start()
        logger.info("PredictionBridge started")
    
    def stop(self):
        """Stop consuming events"""
        self.consumer.stop()
        logger.info("PredictionBridge stopped")
    
    # PART 3: Gesture Mapping & Prediction Processing
    def _process_prediction(self, event: Dict[str, Any]):
        """
        Process gesture prediction from inference.
        
        Flow:
        1. Extract and validate confidence
        2. Deduplicate rapid predictions
        3. Map gesture label to text
        4. Add to buffer
        5. Send output
        6. Provide feedback
        """
        label = event.get('label', '?')
        confidence = event.get('confidence', 0.0)
        timestamp = event.get('timestamp', time.time())
        
        # Step 1: Validate confidence
        if confidence < CONFIDENCE_THRESHOLD:
            logger.debug(f"Skipping low confidence: {label} ({confidence:.2f})")
            return
        
        # Step 2: Deduplicate (skip if same label within 100ms)
        time_since_last = timestamp - self.state.last_emission_time
        if label == self.state.last_label and time_since_last < 0.1:
            logger.debug(f"Duplicate detected: {label}")
            return
        
        # Step 3: Map gesture to text
        text = self.gesture_map.get(label, "?")
        
        # Step 4: Add to buffer
        self.state.add_char(text)
        self.state.last_label = label
        self.state.last_emission_time = timestamp
        
        # Step 5: Send output
        self._send_output(text, 'type')
        
        # Step 6: Feedback
        self._speak_prediction(label)
        self._show_notification(f"Predicted: {label}")
        
        logger.info(f"Prediction processed: {label} → {text}")
    
    # PART 4: Token Processing
    def _process_token(self, event: Dict[str, Any]):
        """
        Process special tokens from inference.
        
        BACKSPACE_TOKEN: Delete last character
        COMMIT_TOKEN: Emit word with space, ready for next word
        """
        token = event.get('token', '?')
        timestamp = event.get('timestamp', time.time())
        
        if token == BACKSPACE_TOKEN:
            if self.state.current_buffer:
                self.state.backspace(count=1)
                self._send_output("backspace", 'action')
            logger.info(f"Backspace processed. Buffer: {self.state.current_buffer}")
        
        elif token == COMMIT_TOKEN:
            emitted = self.state.emit_word(add_space=True)
            if emitted:
                self._send_output(emitted, 'type')
            logger.info(f"Commit processed. Emitted: {emitted.strip()}")
            self._show_notification(f"Word: {emitted.strip()}")
    
    # PART 5: Output Handler
    def _send_output(self, content: str, action: str):
        """
        Send text/action to UI automation layer.
        
        Formats command and routes to ui_automation module.
        
        Args:
            content: Text to type or action to perform
            action: 'type', 'backspace', 'commit'
        """
        command = {
            'action': action,
            'content': content,
            'timestamp': time.time()
        }
        
        try:
            if UI_AUTOMATION_AVAILABLE:
                send_to_ui(command)
                logger.debug(f"Output sent: {action} = {content}")
            else:
                logger.warning("ui_automation module not available - output not sent")
                logger.info(f"[OUTPUT] {action}: {content}")
        except Exception as e:
            logger.error(f"Failed to send output: {e}")
    
    # PART 6: Feedback Systems
    def _speak_prediction(self, label: str):
        """
        Optional: Speak prediction aloud using TTS.
        
        Args:
            label: Gesture label to speak
        """
        if not TTS_AVAILABLE:
            return
        
        try:
            engine = pyttsx3.init()
            engine.say(label)
            engine.runAndWait()
            logger.debug(f"Spoke: {label}")
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
    
    def _show_notification(self, message: str):
        """
        Optional: Show Windows notification.
        
        Args:
            message: Notification message
        """
        if not TOAST_AVAILABLE:
            return
        
        try:
            notifier = ToastNotifier()
            notifier.show_toast("SignBridge", message, duration=1)
            logger.debug(f"Notification: {message}")
        except Exception as e:
            logger.warning(f"Notification failed: {e}")


class QueueConsumerThread:
    """
    Non-blocking queue consumer that listens to predictions from phase4_inference.
    
    Runs in background thread with timeout-based polling.
    """
    
    def __init__(self, input_queue: Queue, queue_timeout: float = 0.1):
        """
        Initialize queue consumer.
        
        Args:
            input_queue: Queue from phase4_inference
            queue_timeout: Timeout for Queue.get() in seconds
        """
        self.input_queue = input_queue
        self.queue_timeout = queue_timeout
        self.running = False
        self.thread = None
        
        # Handlers can be set by PredictionBridge
        self.on_prediction = None
        self.on_token = None
        
        logger.info(f"QueueConsumerThread initialized with timeout={queue_timeout}s")
    
    def start(self):
        """Start consumer thread in background (daemon mode)"""
        if self.running:
            logger.warning("Consumer already running")
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name="PredictionBridgeConsumer"
        )
        self.thread.start()
        logger.info("Queue consumer thread started")
    
    def stop(self):
        """Gracefully stop consumer thread"""
        if not self.running:
            logger.warning("Consumer not running")
            return
        
        logger.info("Stopping queue consumer thread...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("Consumer thread did not stop cleanly")
            else:
                logger.info("Consumer thread stopped")
    
    def _consume_loop(self):
        """Main consumer loop - continuously checks queue with timeout"""
        logger.info("Consumer loop started")
        
        while self.running:
            try:
                event = self.input_queue.get(timeout=self.queue_timeout)
                logger.debug(f"Event received: type={event.get('type')}")
                self._handle_event(event)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Consumer error: {e}", exc_info=True)
                continue
        
        logger.info("Consumer loop stopped")
    
    def _handle_event(self, event: Dict[str, Any]):
        """Route event to appropriate handler"""
        event_type = event.get('type')
        
        if event_type == 'prediction' and self.on_prediction:
            self.on_prediction(event)
        elif event_type == 'token' and self.on_token:
            self.on_token(event)
        else:
            logger.warning(f"Unhandled event type: {event_type}")
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.input_queue.qsize()
    
    def is_running(self) -> bool:
        """Check if consumer is running"""
        return self.running


def main():
    """
    Standalone test: Complete prediction bridge with all parts.
    
    Tests:
    1. BridgeState operations
    2. Queue consumer thread
    3. Prediction processing (Part 3)
    4. Token processing (Part 4)
    5. Output handling (Part 5)
    """
    print("\n" + "="*80)
    print("PREDICTION BRIDGE - COMPLETE IMPLEMENTATION TEST")
    print("="*80 + "\n")
    
    # Test 1: BridgeState
    print("[TEST 1] BridgeState Operations")
    print("-" * 80)
    
    state = BridgeState()
    for char in 'hello':
        state.add_char(char)
    print(f"Buffer: '{state.current_buffer}'")
    
    emitted = state.emit_word(add_space=True)
    print(f"Emitted: '{emitted}'")
    print(f"History: {state.output_history}")
    print()
    
    # Test 2: Prediction Bridge with Queue
    print("[TEST 2] Full Prediction Bridge with Queue")
    print("-" * 80)
    
    test_queue = Queue()
    bridge = PredictionBridge(test_queue)
    
    print("Starting bridge...")
    bridge.start()
    
    # Test events
    test_events = [
        {'type': 'prediction', 'label': 'A', 'confidence': 0.92, 'timestamp': time.time()},
        {'type': 'prediction', 'label': 'B', 'confidence': 0.88, 'timestamp': time.time()},
        {'type': 'token', 'token': COMMIT_TOKEN, 'timestamp': time.time()},
        {'type': 'prediction', 'label': 'C', 'confidence': 0.95, 'timestamp': time.time()},
        {'type': 'token', 'token': BACKSPACE_TOKEN, 'timestamp': time.time()},
        {'type': 'prediction', 'label': 'D', 'confidence': 0.91, 'timestamp': time.time()},
        {'type': 'token', 'token': COMMIT_TOKEN, 'timestamp': time.time()},
    ]
    
    print("\nSending test events...")
    for i, event in enumerate(test_events):
        event_label = event.get('label', event.get('token', 'unknown'))
        print(f"  Event {i+1}: {event['type']:11} = {event_label}")
        test_queue.put(event)
        time.sleep(0.15)
    
    print("\nProcessing...")
    time.sleep(1.5)
    
    print("\nFinal state:")
    stats = bridge.state.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nQueue size: {bridge.consumer.get_queue_size()}")
    
    print("\nStopping bridge...")
    bridge.stop()
    
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
