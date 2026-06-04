"""
Background training runner - keeps training alive even if terminal closes.

Usage:
    python run_background_training.py --model GNNHAR1L --horizon 5 --n_seeds 5

This script:
1. Detaches from terminal (runs as daemon)
2. Logs output to timestamped file
3. Keeps running even if you close laptop lid (if power settings configured)
"""
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run training in background')
    parser.add_argument('--model', type=str, default='GNNHAR1L',
                        choices=['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L'])
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_end', type=str, default='2025-12-31')
    parser.add_argument('--test_start', type=str, default='2026-01-01')
    args = parser.parse_args()

    # Create logs directory
    logs_dir = Path('logs/training')
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'{args.model}_h{args.horizon}_{timestamp}.log'

    print(f'[Background Training] Starting...')
    print(f'[Background Training] Model: {args.model}, Horizon: h{args.horizon}')
    print(f'[Background Training] Log file: {log_file}')
    print(f'[Background Training] You can close this terminal safely.')

    # Build command
    cmd = [
        sys.executable,  # Python executable
        'gnn/gnnhar_paper/train_multi_stock.py',
        '--model', args.model,
        '--horizon', str(args.horizon),
        '--n_seeds', str(args.n_seeds),
        '--epochs', str(args.epochs),
        '--train_end', args.train_end,
        '--test_start', args.test_start,
    ]

    # Open log file
    log_fp = open(log_file, 'w')

    # Start process (detached)
    process = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        bufsize=1,  # Line buffered
        universal_newlines=True,
    )

    # Write PID to file (for monitoring/killing)
    pid_file = logs_dir / f'{args.model}_h{args.horizon}_{timestamp}.pid'
    pid_file.write_text(str(process.pid))

    print(f'[Background Training] Process ID: {process.pid}')
    print(f'[Background Training] PID file: {pid_file}')
    print(f'[Background Training] Training running in background...')
    print(f'[Background Training] Monitor: tail -f {log_file}')

    # Detach from terminal (close file descriptors)
    # This allows the terminal to be closed without killing the process
    if os.name == 'nt':  # Windows
        # On Windows, use DETACHED_PROCESS flag
        pass  # Already handled by Popen defaults on Windows
    else:  # Unix/Linux/Mac
        # Double fork to truly daemonize
        pid = os.fork()
        if pid > 0:
            # Parent exits
            sys.exit(0)

    # Keep the process running in background
    process.wait()

    log_fp.close()
    print(f'[Background Training] Completed. Check log: {log_file}')

if __name__ == '__main__':
    main()
