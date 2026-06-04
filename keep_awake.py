"""
Keep system awake while training runs.

This script prevents Windows from going to sleep by:
1. Pressing Scroll Lock key periodically (harmless, keeps system awake)
2. Can be minimized to tray

Usage:
    python keep_awake.py &
    python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --horizon 5
"""
import time
import ctypes
import threading

def keep_awake(interval=60):
    """
    Prevent system sleep by pressing Scroll Lock periodically.

    Args:
        interval: Seconds between key presses (default 60)
    """
    print(f'[Keep Awake] Running - Press Ctrl+C to stop')
    print(f'[Keep Awake] Presses Scroll Lock every {interval}s')

    try:
        while True:
            # Press Scroll Lock (harmless, keeps system awake)
            ctypes.windll.user32.keybd_event(0x91, 0, 0, 0)  # Scroll Lock down
            ctypes.windll.user32.keybd_event(0x91, 0, 2, 0)  # Scroll Lock up
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\n[Keep Awake] Stopped')

if __name__ == '__main__':
    keep_awake()
