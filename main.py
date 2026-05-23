"""
SignBridge - Main Entry Point

A complete real-time sign language detection pipeline that converts
hand gestures into typed text.

Usage:
    python main.py

Controls:
    - Press Q in the inference window to quit
    - Press SPACE to pause/resume inference
    - Ctrl+C to emergency shutdown

Features:
    - Real-time webcam gesture recognition
    - Intelligent character buffering
    - Automatic word emission
    - Keyboard simulation to any active window
    - TTS feedback (optional)
    - Process health monitoring & auto-restart
"""

import os
import sys
import signal
import logging
from pathlib import Path

# Add src to path for imports
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Disable logging output (no file or console logging)
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

# Import core components
try:
    from background_service import BackgroundService
    from config import HEALTH_CHECK_INTERVAL
    logger.info("Core imports successful")
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    sys.exit(1)


class SignBridgeApplication:
    """
    Main application class that orchestrates the entire pipeline.
    
    Responsibilities:
    - Initialize the background service
    - Handle startup and shutdown
    - Manage signal interrupts
    - Provide user interface feedback
    """
    
    def __init__(self):
        """Initialize the application"""
        self.service = None
        self.running = False
        logger.info("SignBridgeApplication initialized")
    
    def print_header(self):
        """Print startup header"""
        print("\n" + "="*80)
        print("|" + " "*78 + "|")
        print("|" + "SIGNBRIDGE - Sign Language Detection Pipeline".center(78) + "|")
        print("|" + " "*78 + "|")
        print("="*80)
        print()
        print("Starting real-time gesture recognition system...")
        print()
    
    def print_instructions(self):
        """Print user instructions"""
        print("="*80)
        print("CONTROLS & INSTRUCTIONS")
        print("="*80)
        print()
        print("Inference Window (Webcam):")
        print("  - Press Q to quit inference")
        print("  - Press SPACE to pause/resume")
        print()
        print("Pipeline Status:")
        print("  - Watch the terminal for process health updates")
        print("  - Logs are saved to: logs/")
        print()
        print("Optional Features:")
        print("  - TTS feedback: Speaks recognized gestures (if pyttsx3 available)")
        print("  - Notifications: Windows toasts show predictions (if win10toast available)")
        print("  - Keyboard: Types to active window (if pyautogui available)")
        print()
        print("Emergency Stop:")
        print("  - Press Ctrl+C to gracefully shutdown all processes")
        print()
        print("="*80)
        print()
    
    def print_status(self):
        """Print current pipeline status"""
        if self.service and self.service.state:
            stats = self.service.state.get_stats()
            print()
            print("-"*80)
            print("PIPELINE STATUS")
            print("-"*80)
            print(f"  Running: {stats.get('running')}")
            print(f"  Uptime: {stats.get('uptime_seconds'):.1f}s")
            print(f"  Inference Process: {'✓ ALIVE' if stats.get('inference_alive') else '✗ STOPPED'}")
            print(f"  Bridge Process: {'✓ ALIVE' if stats.get('bridge_alive') else '✗ STOPPED'}")
            print(f"  Queue Size: {stats.get('queue_size')} items")
            print(f"  Inference Restarts: {stats.get('inference_restarts')}")
            print(f"  Bridge Restarts: {stats.get('bridge_restarts')}")
            print("-"*80)
            print()
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            """Handle interrupt signals"""
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name} signal - initiating graceful shutdown...")
            print(f"\n\n[*] Received {sig_name} - Shutting down gracefully...\n")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers registered")
    
    def startup(self):
        """Start the background service"""
        try:
            logger.info("Starting BackgroundService...")
            self.service = BackgroundService()
            self.service.start()
            self.running = True
            logger.info("BackgroundService started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start service: {e}", exc_info=True)
            return False
    
    def run(self):
        """Run the main application loop"""
        self.print_header()
        self.print_instructions()
        
        logger.info("="*80)
        logger.info("SignBridge Application Starting")
        logger.info("="*80)
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()
        
        # Start the service
        if not self.startup():
            print("\n[ERROR] Failed to start SignBridge service. Check logs for details.")
            sys.exit(1)
        
        print("[✓] SignBridge service started successfully!")
        print()
        print("Gesture Recognition Running - Wave your hands at the camera!")
        print("(Check the inference window for hand detection feedback)")
        print()
        
        # Main loop - monitor service health
        try:
            import time
            iteration = 0
            while self.running:
                try:
                    time.sleep(HEALTH_CHECK_INTERVAL)
                    iteration += 1
                    
                    # Print status every N iterations (e.g., every 15 seconds if HEALTH_CHECK_INTERVAL=5)
                    if iteration % 3 == 0:
                        self.print_status()
                    
                    # Check if service is still alive
                    if self.service and self.service.state:
                        if not self.service.state.is_running():
                            logger.warning("Service is no longer running")
                            break
                
                except KeyboardInterrupt:
                    raise  # Re-raise to be caught by outer handler
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    continue
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received in main loop")
    
    def shutdown(self):
        """Shutdown the application gracefully"""
        if not self.running:
            return
        
        logger.info("Initiating application shutdown...")
        self.running = False
        
        if self.service:
            try:
                logger.info("Stopping BackgroundService...")
                self.service.stop()
                logger.info("BackgroundService stopped")
            except Exception as e:
                logger.error(f"Error stopping service: {e}", exc_info=True)
        
        print("\n[✓] SignBridge shutdown complete")
        print("All processes terminated gracefully")
        print()
        logger.info("SignBridge Application Shutdown Complete")


def main():
    """Main entry point"""
    try:
        app = SignBridgeApplication()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n[ERROR] Unexpected error: {e}")
        print("Check logs for details")
        sys.exit(1)
    finally:
        logger.info("="*80)
        logger.info("Main process exited")


if __name__ == "__main__":
    main()
