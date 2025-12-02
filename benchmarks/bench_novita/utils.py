# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for multi-LLM benchmarking."""

import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List


class ProcessManager:
    """Manages subprocess lifecycle and cleanup."""

    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Register signal handlers for clean shutdown."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle signals like SIGTERM and SIGINT."""
        print(f"\n\n ERROR: Received signal {signum}. Cleaning up...")
        self.cleanup_all()
        sys.exit(1)

    def register(self, process: subprocess.Popen):
        """Register a process for tracking."""
        self.processes.append(process)

    def cleanup_all(self):
        """Kill all tracked processes."""
        if not self.processes:
            return

        print("Cleaning up all processes...")

        # First, try graceful termination
        for p in self.processes:
            if p.poll() is None:  # Process still running
                try:
                    p.terminate()
                except Exception as e:
                    print(f"Error terminating PID {p.pid}: {e}")

        # Wait a bit for graceful shutdown
        time.sleep(2)

        # Force kill any remaining processes
        for p in self.processes:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception as e:
                    print(f"Error killing PID {p.pid}: {e}")

        # Wait for all to finish
        for p in self.processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Warning: Process PID {p.pid} did not exit after kill signal")
            except Exception as e:
                print(f"Error waiting for PID {p.pid}: {e}")

        print("All processes cleanup complete")


def print_log_tail(log_file: Path, num_lines: int = 30):
    """Print the last N lines of a log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-num_lines:]:
                print(f"      {line.rstrip()}")
    except Exception as e:
        print(f"   Could not read log file: {e}")


def check_server_health(url: str, timeout: int = 5) -> bool:
    """Check if a server is healthy by making an HTTP request."""
    import requests
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def format_duration(seconds: float) -> str:
    """Format duration in seconds to the best unit (s, m, h)."""
    if seconds < 60:
        # For seconds: use integer if whole number
        if abs(seconds - round(seconds)) < 0.1:
            return f"{int(round(seconds))}s"
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        # For minutes: use integer if whole number, otherwise 1 decimal
        minutes = seconds / 60
        if abs(minutes - round(minutes)) < 0.01:
            return f"{int(round(minutes))}m"
        return f"{minutes:.1f}m"
    else:
        # For hours: use integer if whole number, otherwise 2 decimals
        hours = seconds / 3600
        if abs(hours - round(hours)) < 0.01:
            return f"{int(round(hours))}h"
        return f"{hours:.2f}h"


def calculate_benchmark_durations(benchmark_config: dict) -> tuple[str, str]:
    """
    Calculate and format original trace duration and actual replay duration.

    Args:
        benchmark_config: Dictionary containing 'start_timestamp', 'end_timestamp', and 'slowdown_factor'

    Returns:
        Tuple of (original_duration_str, actual_trace_time_str)
    """
    from datetime import datetime

    # Parse timestamps
    start_timestamp = datetime.strptime(
        benchmark_config['start_timestamp'],
        '%Y-%m-%d %H:%M:%S.%f'
    )
    end_timestamp = datetime.strptime(
        benchmark_config['end_timestamp'],
        '%Y-%m-%d %H:%M:%S.%f'
    )

    # Calculate original duration
    original_duration = (end_timestamp - start_timestamp).total_seconds()

    # Calculate actual trace time considering slowdown_factor
    slowdown_factor = benchmark_config['slowdown_factor']
    actual_trace_time = original_duration * slowdown_factor

    # Format durations with appropriate units
    original_duration_str = format_duration(original_duration)
    actual_trace_time_str = format_duration(actual_trace_time)

    return original_duration_str, actual_trace_time_str
