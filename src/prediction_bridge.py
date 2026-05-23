"""
Prediction Bridge - Part 1: Queue Consumer Thread

Listens to Queue from phase4_inference.py and routes events for processing.
Implements non-blocking consumer with timeout-based polling.
"""

import logging
import threading
import time
from queue import Queue, Empty
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class QueueConsumerThread:
    """
    Non-blocking queue consumer that listens to predictions from phase4_inference.
    
    Pattern:
    - Runs in background thread (daemon mode)
    - Non-blocking with configurable timeout
    - Routes events to appropriate handlers
    - Graceful shutdown support
    """
    
    def __init__(self, input_queue: Queue, queue_timeout: float = 0.1):
        """
        Initialize queue consumer.
        
        Args:
            input_queue: Queue from phase4_inference containing prediction events
            queue_timeout: Timeout for Queue.get() in seconds (default 0.1s)
        """
        self.input_queue = input_queue
        self.queue_timeout = queue_timeout
        self.running = False
        self.thread = None
        
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
        
        # Wait for thread to finish (up to 5 seconds)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("Consumer thread did not stop cleanly")
            else:
                logger.info("Consumer thread stopped")
    
    def _consume_loop(self):
        """
        Main consumer loop - continuously checks queue with timeout.
        
        Flow:
        1. Call Queue.get(timeout=X) - blocks for X seconds max
        2. If Empty exception raised - timeout occurred, loop continues
        3. Process event via _handle_event()
        4. Check self.running flag to allow graceful shutdown
        """
        logger.info("Consumer loop started")
        
        while self.running:
            try:
                # Non-blocking get with timeout
                event = self.input_queue.get(timeout=self.queue_timeout)
                
                # Successfully received event
                logger.debug(f"Event received: type={event.get('type')}")
                self._handle_event(event)
                
            except Empty:
                # No event within timeout - continue loop
                # This allows checking self.running flag periodically
                continue
                
            except Exception as e:
                # Unexpected error - log and continue
                logger.error(f"Unexpected error in consumer loop: {e}", exc_info=True)
                continue
        
        logger.info("Consumer loop stopped")
    
    def _handle_event(self, event: Dict[str, Any]):
        """
        Route event to appropriate handler based on type.
        
        Event types:
        - 'prediction': gesture label with confidence
        - 'token': special token (BACKSPACE_TOKEN, COMMIT_TOKEN)
        
        Args:
            event: Dictionary with 'type' and event-specific fields
        """
        event_type = event.get('type')
        
        if event_type == 'prediction':
            self._handle_prediction(event)
            
        elif event_type == 'token':
            self._handle_token(event)
            
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def _handle_prediction(self, event: Dict[str, Any]):
        """
        Handle prediction event from inference.
        
        Expected fields:
        - label: gesture label (str)
        - confidence: confidence score (float)
        - timestamp: event timestamp (float)
        
        Part 2 will implement actual processing here.
        """
        label = event.get('label', '?')
        confidence = event.get('confidence', 0.0)
        timestamp = event.get('timestamp', time.time())
        
        logger.info(f"Prediction: label={label}, conf={confidence:.2f}, time={timestamp}")
        
        # TODO: Part 2 - Process prediction
        # - Validate confidence threshold
        # - Deduplicate
        # - Map gesture to text
        # - Add to buffer
        # - Send output
    
    def _handle_token(self, event: Dict[str, Any]):
        """
        Handle token event (BACKSPACE_TOKEN, COMMIT_TOKEN).
        
        Expected fields:
        - token: special token string (str)
        - timestamp: event timestamp (float)
        
        Part 4 will implement actual processing here.
        """
        token = event.get('token', '?')
        timestamp = event.get('timestamp', time.time())
        
        logger.info(f"Token: {token}, time={timestamp}")
        
        # TODO: Part 4 - Process token
        # - BACKSPACE_TOKEN: delete from buffer
        # - COMMIT_TOKEN: emit word and add space
    
    def get_queue_size(self) -> int:
        """Get current queue size (for diagnostics)"""
        return self.input_queue.qsize()
    
    def is_running(self) -> bool:
        """Check if consumer is running"""
        return self.running


def main():
    """
    Standalone test: consume predictions from a test queue.
    
    For testing Part 1 only.
    """
    # Create test queue and consumer
    test_queue = Queue()
    consumer = QueueConsumerThread(test_queue, queue_timeout=0.1)
    
    logger.info("Starting queue consumer test...")
    consumer.start()
    
    # Simulate events from inference
    test_events = [
        {'type': 'prediction', 'label': 'A', 'confidence': 0.92, 'timestamp': time.time()},
        {'type': 'prediction', 'label': 'B', 'confidence': 0.88, 'timestamp': time.time()},
        {'type': 'token', 'token': '__COMMIT__', 'timestamp': time.time()},
        {'type': 'prediction', 'label': 'C', 'confidence': 0.95, 'timestamp': time.time()},
    ]
    
    # Put test events into queue
    for event in test_events:
        test_queue.put(event)
        time.sleep(0.2)  # Small delay between events
    
    # Let consumer process for a bit
    time.sleep(1.0)
    
    logger.info(f"Queue size: {consumer.get_queue_size()}")
    logger.info("Stopping consumer...")
    consumer.stop()
    
    logger.info("Test complete")


if __name__ == "__main__":
    main()
