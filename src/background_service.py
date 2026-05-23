"""
Background Service - Step 3

Orchestrates and manages the entire inference pipeline:
1. Spawns phase4_inference.py subprocess (produces predictions)
2. Spawns prediction_bridge.py subprocess (consumes predictions, outputs to UI)
3. Manages shared Queue between subprocesses
4. Monitors subprocess health and auto-restarts on crash
5. Aggregates logs from all components
6. Handles graceful shutdown on Ctrl+C
"""

import logging
import multiprocessing
import signal
import sys
import threading
import time
from datetime import datetime
from multiprocessing import Queue, Process, Manager
from typing import Dict, Optional, Any

# Disable logging output (no file or console logging)
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

# Try to import psutil for enhanced process monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available - basic process monitoring only")

# Import config
try:
    from config import (
        HEALTH_CHECK_INTERVAL,
        PROCESS_RESTART_LIMIT,
        RESTART_BACKOFF_SECONDS,
        QUEUE_STALL_TIMEOUT,
        QUEUE_OVERFLOW_THRESHOLD,
        PROCESS_SHUTDOWN_TIMEOUT,
    )
except ImportError as e:
    logger.warning(f"Config import failed: {e} - using defaults")
    # Defaults
    HEALTH_CHECK_INTERVAL = 5
    PROCESS_RESTART_LIMIT = 3
    RESTART_BACKOFF_SECONDS = 1
    QUEUE_STALL_TIMEOUT = 30
    QUEUE_OVERFLOW_THRESHOLD = 100
    PROCESS_SHUTDOWN_TIMEOUT = 5

# Import subprocess modules
try:
    from phase4_inference import run_webcam_inference, load_artifacts
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    logger.warning("phase4_inference module not available")

try:
    from prediction_bridge import PredictionBridge
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    logger.warning("prediction_bridge module not available")


class BackgroundServiceState:
    """
    Track background service state and health metrics.
    """
    
    def __init__(self):
        """Initialize service state"""
        self.running = False
        self.inference_process: Optional[Process] = None
        self.bridge_process: Optional[Process] = None
        self.manager: Optional[Manager] = None
        self.shared_queue: Optional[Queue] = None
        self.start_time: float = 0.0
        self.process_restarts: Dict[str, int] = {
            'inference': 0,
            'bridge': 0
        }
        self.last_health_check: float = 0.0
        self.last_queue_size: int = 0
        self.last_queue_activity: float = 0.0
    
    def is_running(self) -> bool:
        """Check if service is running"""
        return self.running
    
    def is_process_alive(self, process_name: str) -> bool:
        """
        Check if subprocess is alive.
        
        Args:
            process_name: 'inference' or 'bridge'
        
        Returns:
            True if process is alive
        """
        if process_name == 'inference':
            process = self.inference_process
        elif process_name == 'bridge':
            process = self.bridge_process
        else:
            return False
        
        if process is None:
            return False
        
        # Check if process is still alive
        return process.is_alive()
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        try:
            queue_size = self.shared_queue.qsize() if self.shared_queue else 0
        except Exception:
            queue_size = 0
        
        return {
            'running': self.running,
            'uptime_seconds': self.get_uptime(),
            'inference_alive': self.is_process_alive('inference'),
            'bridge_alive': self.is_process_alive('bridge'),
            'inference_restarts': self.process_restarts['inference'],
            'bridge_restarts': self.process_restarts['bridge'],
            'queue_size': queue_size,
            'last_health_check': datetime.fromtimestamp(self.last_health_check).isoformat(),
        }


class ProcessManager:
    """
    Create and manage subprocesses for inference and bridge.
    
    Uses multiprocessing.Process to enable natural queue sharing.
    """
    
    def __init__(self, state: BackgroundServiceState):
        """
        Initialize process manager.
        
        Args:
            state: BackgroundServiceState instance
        """
        self.state = state
        logger.info("ProcessManager initialized")
    
    def start_inference(self, queue: Queue) -> Process:
        """
        Start phase4_inference.py subprocess.
        
        Args:
            queue: Multiprocessing Queue to pass to subprocess
        
        Returns:
            Process handle
        """
        try:
            logger.info("Starting inference subprocess...")
            
            # Create process that runs inference with queue
            process = Process(
                target=self._run_inference_wrapper,
                args=(queue,),
                name="InferenceProcess"
            )
            process.daemon = False
            process.start()
            
            logger.info(f"Inference subprocess started (PID: {process.pid})")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start inference subprocess: {e}", exc_info=True)
            return None
    
    def start_bridge(self, queue: Queue) -> Process:
        """
        Start prediction_bridge.py subprocess.
        
        Args:
            queue: Multiprocessing Queue to pass to subprocess
        
        Returns:
            Process handle
        """
        try:
            logger.info("Starting bridge subprocess...")
            
            # Create process that runs bridge with queue
            process = Process(
                target=self._run_bridge_wrapper,
                args=(queue,),
                name="BridgeProcess"
            )
            process.daemon = False
            process.start()
            
            logger.info(f"Bridge subprocess started (PID: {process.pid})")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start bridge subprocess: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _run_inference_wrapper(queue: Queue):
        """
        Wrapper to run inference in subprocess with queue.
        
        Args:
            queue: Shared queue for output
        """
        try:
            logger.info("Inference subprocess: Starting inference loop")
            if not INFERENCE_AVAILABLE:
                logger.error("phase4_inference not available")
                return
            
            model, encoder = load_artifacts()
            run_webcam_inference(model, encoder, output_queue=queue)
            
        except Exception as e:
            logger.error(f"Inference subprocess error: {e}", exc_info=True)
    
    @staticmethod
    def _run_bridge_wrapper(queue: Queue):
        """
        Wrapper to run bridge in subprocess with queue.
        
        Args:
            queue: Shared queue for input
        """
        try:
            logger.info("Bridge subprocess: Starting bridge consumer")
            if not BRIDGE_AVAILABLE:
                logger.error("prediction_bridge not available")
                return
            
            bridge = PredictionBridge(queue)
            bridge.start()
            
            # Keep the process alive while consumer thread runs
            # The consumer is a daemon thread, so it will keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                bridge.stop()
            
        except Exception as e:
            logger.error(f"Bridge subprocess error: {e}", exc_info=True)
    
    def terminate_process(self, process: Process, name: str, timeout: float = PROCESS_SHUTDOWN_TIMEOUT):
        """
        Gracefully terminate a subprocess.
        
        Args:
            process: Process handle
            name: Process name for logging
            timeout: Seconds to wait before force kill
        """
        if process is None:
            return
        
        try:
            logger.info(f"Terminating {name} (PID: {process.pid})...")
            
            # Send terminate signal (graceful shutdown)
            process.terminate()
            
            # Wait for graceful shutdown
            process.join(timeout=timeout)
            
            if process.is_alive():
                # Force kill
                logger.warning(f"{name} did not terminate gracefully, force killing...")
                process.kill()
                process.join()
                logger.info(f"{name} force killed")
            else:
                logger.info(f"{name} terminated gracefully")
                
        except Exception as e:
            logger.error(f"Error terminating {name}: {e}")
    
    def get_process_status(self, process: Process, name: str) -> str:
        """
        Get process status.
        
        Args:
            process: Process handle
            name: Process name for logging
        
        Returns:
            'running', 'stopped', or 'crashed'
        """
        if process is None:
            return 'stopped'
        
        if process.is_alive():
            return 'running'
        else:
            exit_code = process.exitcode
            if exit_code == 0:
                return 'stopped'
            else:
                logger.warning(f"{name} crashed with exit code {exit_code}")
                return 'crashed'


class HealthMonitor(threading.Thread):
    """
    Monitor subprocess health and restart if needed.
    
    Runs in background thread, checks health every 5 seconds.
    Uses multiprocessing.Process which has is_alive() and exitcode attributes.
    """
    
    def __init__(self, state: BackgroundServiceState, process_manager: ProcessManager):
        """
        Initialize health monitor.
        
        Args:
            state: BackgroundServiceState instance
            process_manager: ProcessManager instance
        """
        super().__init__(daemon=True, name="HealthMonitor")
        self.state = state
        self.process_manager = process_manager
        self.running = False
        
        logger.info("HealthMonitor initialized")
    
    def run(self):
        """Main health check loop"""
        logger.info("Health monitor started")
        self.running = True
        
        while self.running and self.state.running:
            try:
                # Check inference process
                if not self.state.is_process_alive('inference'):
                    self._handle_process_crash('inference')
                
                # Check bridge process
                if not self.state.is_process_alive('bridge'):
                    self._handle_process_crash('bridge')
                
                # Check queue health
                self._check_queue_health()
                
                # Update last health check time
                self.state.last_health_check = time.time()
                
                # Sleep before next check
                time.sleep(HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}", exc_info=True)
                time.sleep(1)
        
        logger.info("Health monitor stopped")
    
    def _handle_process_crash(self, process_name: str):
        """
        Handle crashed subprocess.
        
        Args:
            process_name: 'inference' or 'bridge'
        """
        restarts = self.state.process_restarts[process_name]
        
        logger.error(f"{process_name} process crashed (restart #{restarts + 1})")
        
        if restarts >= PROCESS_RESTART_LIMIT:
            logger.critical(
                f"{process_name} reached max restarts ({PROCESS_RESTART_LIMIT}), "
                "not restarting. Manual intervention required."
            )
            return
        
        # Calculate backoff
        backoff = min(RESTART_BACKOFF_SECONDS * (2 ** restarts), 8)
        logger.info(f"Waiting {backoff}s before restart...")
        time.sleep(backoff)
        
        # Restart process
        try:
            if process_name == 'inference':
                self.state.inference_process = self.process_manager.start_inference(self.state.shared_queue)
            elif process_name == 'bridge':
                self.state.bridge_process = self.process_manager.start_bridge(self.state.shared_queue)
            
            self.state.process_restarts[process_name] += 1
            logger.info(f"{process_name} restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart {process_name}: {e}", exc_info=True)
    
    def _check_queue_health(self):
        """Check queue health and log warnings"""
        if self.state.shared_queue is None:
            return
        
        try:
            queue_size = self.state.shared_queue.qsize()
            
            # Check for overflow
            if queue_size > QUEUE_OVERFLOW_THRESHOLD:
                logger.warning(f"Queue overflow: {queue_size} items (threshold: {QUEUE_OVERFLOW_THRESHOLD})")
            
            # Check for stall
            if queue_size != self.state.last_queue_size:
                self.state.last_queue_activity = time.time()
            else:
                time_since_activity = time.time() - self.state.last_queue_activity
                if time_since_activity > QUEUE_STALL_TIMEOUT:
                    logger.warning(f"Queue stalled for {time_since_activity:.1f}s")
            
            self.state.last_queue_size = queue_size
            
        except Exception as e:
            logger.debug(f"Error checking queue health: {e}")
    
    def stop(self):
        """Stop health monitor"""
        logger.info("Stopping health monitor...")
        self.running = False


class LogAggregator(threading.Thread):
    """
    Aggregate logs from all components.
    
    With multiprocessing.Process, subprocesses share logging configuration
    so this thread primarily monitors queue and service health.
    """
    
    def __init__(self, state: BackgroundServiceState):
        """
        Initialize log aggregator.
        
        Args:
            state: BackgroundServiceState instance
        """
        super().__init__(daemon=True, name="LogAggregator")
        self.state = state
        self.running = False
        
        logger.info("LogAggregator initialized")
    
    def run(self):
        """Main log aggregation loop"""
        logger.info("Log aggregator started (monitoring queue health)")
        self.running = True
        
        while self.running and self.state.running:
            try:
                # Get current stats for logging
                if self.state.last_health_check > 0:
                    stats = self.state.get_stats()
                    queue_size = stats.get('queue_size', 0)
                    
                    # Log high queue size warnings
                    if queue_size > QUEUE_OVERFLOW_THRESHOLD:
                        logger.warning(f"Queue backlog: {queue_size} items")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Log aggregator error: {e}", exc_info=True)
                time.sleep(1)
        
        logger.info("Log aggregator stopped")
    
    def stop(self):
        """Stop log aggregator"""
        logger.info("Stopping log aggregator...")
        self.running = False


class BackgroundService:
    """
    Main background service orchestrator.
    
    Manages:
    - Subprocess lifecycle (inference, bridge)
    - Shared Queue IPC
    - Health monitoring and auto-restart
    - Log aggregation
    - Graceful shutdown
    """
    
    def __init__(self):
        """Initialize background service"""
        self.state = BackgroundServiceState()
        self.process_manager = ProcessManager(self.state)
        self.health_monitor: Optional[HealthMonitor] = None
        self.log_aggregator: Optional[LogAggregator] = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("BackgroundService initialized")
    
    def start(self):
        """Start background service"""
        logger.info("="*80)
        logger.info("Starting SignBridge Background Service...")
        logger.info("="*80)
        
        self.state.running = True
        self.state.start_time = time.time()
        
        # Create manager and shared queue
        try:
            self.state.manager = Manager()
            self.state.shared_queue = self.state.manager.Queue()
            logger.info("Manager-backed Queue created (supports process sharing)")
        except Exception as e:
            logger.error(f"Failed to create manager queue: {e}")
            self.stop()
            return
        
        # Start inference subprocess
        self.state.inference_process = self.process_manager.start_inference(self.state.shared_queue)
        if self.state.inference_process is None:
            logger.error("Failed to start inference subprocess")
            self.stop()
            return
        
        time.sleep(0.5)  # Brief delay before starting bridge
        
        # Start bridge subprocess
        self.state.bridge_process = self.process_manager.start_bridge(self.state.shared_queue)
        if self.state.bridge_process is None:
            logger.error("Failed to start bridge subprocess")
            self.stop()
            return
        
        # Start health monitor thread
        self.health_monitor = HealthMonitor(self.state, self.process_manager)
        self.health_monitor.start()
        
        # Start log aggregator thread
        self.log_aggregator = LogAggregator(self.state)
        self.log_aggregator.start()
        
        logger.info("="*80)
        logger.info("Service started successfully")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*80)
    
    def run(self):
        """
        Run service (blocking).
        
        Maintains service loop until stop() is called.
        """
        try:
            while self.state.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop background service gracefully"""
        if not self.state.running:
            logger.debug("Service already stopped")
            return
        
        logger.info("="*80)
        logger.info("Graceful shutdown initiated...")
        logger.info("="*80)
        
        self.state.running = False
        
        # Stop health monitor
        if self.health_monitor:
            self.health_monitor.stop()
            self.health_monitor.join(timeout=2)
        
        # Stop log aggregator
        if self.log_aggregator:
            self.log_aggregator.stop()
            self.log_aggregator.join(timeout=2)
        
        # Terminate subprocesses
        self.process_manager.terminate_process(self.state.inference_process, "Inference")
        self.process_manager.terminate_process(self.state.bridge_process, "Bridge")
        
        # Shutdown manager and queue
        if self.state.shared_queue:
            try:
                self.state.shared_queue.close()
            except Exception:
                pass
            self.state.shared_queue = None

        if self.state.manager:
            try:
                self.state.manager.shutdown()
            except Exception:
                pass
            self.state.manager = None
        
        logger.info("="*80)
        logger.info("Service stopped")
        logger.info(f"Final stats: {self.state.get_stats()}")
        logger.info("="*80)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current service status.
        
        Returns:
            Dictionary with service statistics
        """
        return self.state.get_stats()
    
    def _signal_handler(self, signum, frame):
        """
        Handle system signals (Ctrl+C, kill).
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Signal {signum} received")
        self.stop()
        sys.exit(0)


def main():
    """Entry point for background service"""
    service = BackgroundService()
    service.start()
    service.run()


if __name__ == "__main__":
    main()
