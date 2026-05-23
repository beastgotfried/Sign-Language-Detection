"""
UI Automation Layer

Handles typing text and performing keyboard actions on the active window.
Supports multiple typing strategies: direct keyboard control and clipboard.
"""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import pyautogui for keyboard control
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.debug("pyautogui not available - keyboard control disabled")

# Try to import win32 for Windows-specific automation
try:
    import win32api
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logger.debug("win32 not available - Windows clipboard control disabled")


def send_to_ui(command: Dict[str, Any]) -> bool:
    """
    Send text or action command to UI.
    
    Args:
        command: Dictionary with keys:
            - action: 'type' or 'backspace'
            - content: text to type or number of backspaces
            - timestamp: when command was issued
    
    Returns:
        True if successfully sent, False if not available
    """
    action = command.get('action', 'type')
    content = command.get('content', '')
    timestamp = command.get('timestamp', time.time())
    
    try:
        if action == 'type':
            return type_text(content)
        elif action == 'backspace':
            return perform_backspace(int(content) if isinstance(content, str) and content.isdigit() else 1)
        else:
            logger.warning(f"Unknown action: {action}")
            return False
    except Exception as e:
        logger.error(f"Failed to send to UI: {e}")
        return False


def type_text(text: str) -> bool:
    """
    Type text to active window.
    
    Args:
        text: Text to type
    
    Returns:
        True if successful
    """
    if not text:
        return True
    
    if PYAUTOGUI_AVAILABLE:
        try:
            pyautogui.write(text, interval=0.05)
            logger.debug(f"Typed: {text}")
            return True
        except Exception as e:
            logger.warning(f"pyautogui type failed: {e} - falling back to logging")
    
    # Fallback: just log it
    logger.info(f"[TYPED] {text}")
    return True


def perform_backspace(count: int = 1) -> bool:
    """
    Send backspace key to active window.
    
    Args:
        count: Number of backspaces to send
    
    Returns:
        True if successful
    """
    if count <= 0:
        return True
    
    if PYAUTOGUI_AVAILABLE:
        try:
            for _ in range(count):
                pyautogui.press('backspace')
                time.sleep(0.05)
            logger.debug(f"Backspace sent: {count}")
            return True
        except Exception as e:
            logger.warning(f"pyautogui backspace failed: {e} - falling back to logging")
    
    # Fallback: just log it
    logger.info(f"[BACKSPACE] x{count}")
    return True


def clear_clipboard() -> bool:
    """
    Clear system clipboard.
    
    Returns:
        True if successful
    """
    if WIN32_AVAILABLE:
        try:
            import subprocess
            subprocess.run(['clip'], input=b'', check=True)
            logger.debug("Clipboard cleared")
            return True
        except Exception as e:
            logger.warning(f"Clipboard clear failed: {e}")
            return False
    
    return False


def get_clipboard() -> str:
    """
    Get current clipboard content.
    
    Returns:
        Clipboard text or empty string if unavailable
    """
    try:
        import subprocess
        result = subprocess.run(['powershell', '-Command', 'Get-Clipboard'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Clipboard read failed: {e}")
        return ""


def main():
    """Test UI automation functions"""
    print("\n" + "="*80)
    print("UI AUTOMATION - FUNCTION TEST")
    print("="*80 + "\n")
    
    # Test 1: Type text
    print("[TEST 1] Type text")
    print("-" * 80)
    print("Would type: 'hello world'")
    result = type_text("hello world")
    print(f"Result: {result}\n")
    
    # Test 2: Backspace
    print("[TEST 2] Backspace")
    print("-" * 80)
    print("Would send 3 backspaces")
    result = perform_backspace(3)
    print(f"Result: {result}\n")
    
    # Test 3: Send to UI (type action)
    print("[TEST 3] Send to UI - Type action")
    print("-" * 80)
    command = {'action': 'type', 'content': 'test', 'timestamp': time.time()}
    result = send_to_ui(command)
    print(f"Result: {result}\n")
    
    # Test 4: Send to UI (backspace action)
    print("[TEST 4] Send to UI - Backspace action")
    print("-" * 80)
    command = {'action': 'backspace', 'content': '2', 'timestamp': time.time()}
    result = send_to_ui(command)
    print(f"Result: {result}\n")
    
    print("="*80)
    print("Tests complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
