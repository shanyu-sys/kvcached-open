#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Multi-LLM benchmark orchestrator."""

import argparse
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml
from utils import ProcessManager, calculate_benchmark_durations, check_server_health, print_log_tail

# Environment settings
os.environ["HF_HUB_OFFLINE"] = "0"


@dataclass
class Instance:
    """Configuration for a single LLM instance."""
    name: str
    model_name: str
    port: int
    trace_file: str
    trace_model_filter: Optional[str] = None
    model_path: Optional[str] = None
    gpu_id: Optional[int] = None
    tp: Optional[int] = None  # Tensor parallelism size


def parse_duration(duration_str: str) -> timedelta:
    """
    Parse a duration string into a timedelta object.
    
    Supported formats:
        - "30s" or "30sec" -> 30 seconds
        - "10m" or "10min" -> 10 minutes
        - "1h" or "1hour" -> 1 hour
        - "1.5h" -> 1.5 hours
        - "90" -> 90 seconds (default unit is seconds)
    
    Args:
        duration_str: Duration string to parse
        
    Returns:
        timedelta object representing the duration
        
    Raises:
        ValueError: If the duration string format is invalid
    """
    duration_str = duration_str.strip().lower()
    
    # Pattern to match number (int or float) followed by optional unit
    pattern = r'^(\d+(?:\.\d+)?)\s*(s|sec|seconds?|m|min|minutes?|h|hr|hours?)?$'
    match = re.match(pattern, duration_str)
    
    if not match:
        raise ValueError(f"Invalid duration format: '{duration_str}'. "
                        f"Expected formats: '30s', '10m', '1h', '1.5h', or just a number (seconds)")
    
    value = float(match.group(1))
    unit = match.group(2) or 's'  # Default to seconds
    
    if unit in ('s', 'sec', 'second', 'seconds'):
        return timedelta(seconds=value)
    elif unit in ('m', 'min', 'minute', 'minutes'):
        return timedelta(minutes=value)
    elif unit in ('h', 'hr', 'hour', 'hours'):
        return timedelta(hours=value)
    else:
        return timedelta(seconds=value)


def resolve_end_timestamp(benchmark_config: dict) -> None:
    """
    Resolve end_timestamp from duration if specified.
    
    If 'duration' is specified in benchmark_config, calculate end_timestamp
    as start_timestamp + duration. Modifies benchmark_config in place.
    
    Args:
        benchmark_config: Benchmark configuration dictionary
    """
    if 'duration' in benchmark_config and benchmark_config['duration']:
        start_str = benchmark_config['start_timestamp']
        start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S.%f')
        
        duration = parse_duration(str(benchmark_config['duration']))
        end_dt = start_dt + duration
        
        benchmark_config['end_timestamp'] = end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        print(f"Duration '{benchmark_config['duration']}' resolved to end_timestamp: {benchmark_config['end_timestamp']}")


def load_config(config_path: str = "config.yml") -> dict:
    """
    Load instance configurations from YAML file.
    
    Args:
        config_path: Path to the config file
                         
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve duration to end_timestamp if needed
    if 'benchmark' in config:
        resolve_end_timestamp(config['benchmark'])
    
    return config


def create_log_directories(log_dir: Path) -> tuple[Path, Path]:
    """Create timestamped log directories for servers and benchmarks."""

    server_log_dir = log_dir / "servers"
    benchmark_log_dir = log_dir / "benchmarks"

    server_log_dir.mkdir(parents=True, exist_ok=True)
    benchmark_log_dir.mkdir(parents=True, exist_ok=True)

    return server_log_dir, benchmark_log_dir


def launch_servers(
                   config: dict,
                   instances: list[Instance],
                   log_dir: Path, process_manager: ProcessManager) -> list[Path]:

    print("=" * 80)
    print("LAUNCHING SERVERS")
    print("=" * 80)
    print(f"Server logs directory: {log_dir.absolute()}")
    print()

    processes = []
    log_files = []

    # Launch all servers
    for instance in instances:
        log_file = log_dir / f"server_{instance.name}_port{instance.port}.log"
        log_files.append(log_file)

        # Determine GPU ID
        base_gpu_id = 0
        if "gpu_id" in config["benchmark"]:
            base_gpu_id = config["benchmark"]["gpu_id"]
        elif instance.gpu_id is not None:
            base_gpu_id = instance.gpu_id
        
        # Determine TP size (default to 1 if not specified)
        if "tp" in config["benchmark"]:
            tp_size = config["benchmark"]["tp"]
        elif instance.tp is not None:
            tp_size = instance.tp
        else:
            tp_size = 1
        
        # Set CUDA_VISIBLE_DEVICES: for TP > 1, we need multiple GPUs
        if tp_size > 1:
            gpu_list = [str(base_gpu_id + i) for i in range(tp_size)]
            cuda_visible_devices = ",".join(gpu_list)
        else:
            cuda_visible_devices = str(base_gpu_id)
        
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        })

        # Build command
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model", instance.model_path if instance.model_path else instance.model_name,
            "--port", str(instance.port),
            "--disable-radix-cache",
            "--disable-cuda-graph",
        ]
        
        # Add TP flag if TP > 1
        if tp_size > 1:
            cmd.extend(["--tp", str(tp_size)])

        print(f"  Launching server for {instance.name}")
        print(f"   Port: {instance.port}")
        print(f"   TP: {tp_size}")
        print(f"   GPUs: {cuda_visible_devices}")
        print(f"   Log: {log_file}")

        # Launch process
        with open(log_file, "w") as log_f:
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Started at: {datetime.now()}\n")
            log_f.write("=" * 80 + "\n\n")
            log_f.flush()

            process = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)

        processes.append(process)
        process_manager.register(process)
        print(f"   PID: {process.pid}")
        print()

    # Wait for all servers to be ready
    print("\nWaiting for servers to start...")
    print("-" * 80)

    server_start_time = time.time()

    for idx, instance in enumerate(instances):
        url = f"http://localhost:{instance.port}"
        print(f"\n Checking {instance.name} at {url}...")

        start_launch_time = time.time()
        ready = False
        for attempt in range(60):  # 60 attempts * 2 seconds = 2 minutes timeout
            # Check if process is still running
            if processes[idx].poll() is not None:
                print(f"   ERROR: Server process died (exit code: {processes[idx].returncode})")
                print(f"   Log file: {log_files[idx]}")
                print("   Last 30 lines of log:")
                print_log_tail(log_files[idx], 30)
                process_manager.cleanup_all()
                raise RuntimeError(f"Server for {instance.name} failed to start")

            # Check health
            if check_server_health(url + "/get_model_info"):
                time_taken = time.time() - start_launch_time
                print(f"  Server is UP! (took {time_taken:.2f} seconds)")
                ready = True
                break

            time.sleep(2)

        if not ready:
            print("  ERROR: Server failed to become ready after 60 attempts")
            print(f"   Log file: {log_files[idx]}")
            process_manager.cleanup_all()
            raise RuntimeError(f"Server for {instance.name} did not become ready in time")

    print("\n ALL SERVERS ARE READY! ")
    print(f"  Time taken to start all servers: {time.time() - server_start_time:.2f} seconds")
    print()

    return log_files


def launch_benchmarks(config: dict,
                      dataset_name: str,
                      instances: list[Instance],
                     log_dir: Path, process_manager: ProcessManager) -> bool:
    print("=" * 80)
    print("LAUNCHING BENCHMARK CLIENTS")
    print("=" * 80)
    print(f"Benchmark logs directory: {log_dir.absolute()}")
    print()

    bench_config = config['benchmark']
    processes = []
    log_files = []

    # Launch all benchmarks
    for rank, instance in enumerate(instances):
        log_file = log_dir / f"benchmark_{instance.name}_port{instance.port}.log"
        log_files.append(log_file)

        url = f"http://localhost:{instance.port}"
        cmd = [
            "python3",
            "sglang_bench_serving.py",
            "--backend", "sglang",
            "--base-url", url,
            "--model", instance.model_path if instance.model_path else instance.model_name,
            "--dataset-name", dataset_name,
            "--slowdown-factor", str(bench_config['slowdown_factor']),
            "--start-timestamp", bench_config['start_timestamp'],
            "--end-timestamp", bench_config['end_timestamp'],
        ]

        if dataset_name == "battle":
            cmd.extend(["--trace-model-filter", instance.trace_model_filter])
        
        if dataset_name == "novita":
            cmd.extend(["--dataset-path", instance.trace_file,
            ])

        env = os.environ.copy()
        env.update({
            "MASTER_ADDR": bench_config['torch_master_addr'],
            "MASTER_PORT": bench_config['torch_master_port'],
            "WORLD_SIZE": str(len(instances)),
            "RANK": str(rank),
        })

        print(f"  Launching benchmark for {instance.name}")
        print(f"   Log: {log_file}")

        # Launch process
        with open(log_file, "w") as log_f:
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Started at: {datetime.now()}\n")
            log_f.write(f"Environment: RANK={rank} WORLD_SIZE={len(instances)}\n")
            log_f.write("=" * 80 + "\n\n")
            log_f.flush()

            process = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)

        processes.append(process)
        process_manager.register(process)
        print(f"   PID: {process.pid}")
        print()

    # Wait for all benchmarks to complete
    print("-" * 80)
    print("All benchmark clients launched. Waiting for completion...")
    print("-" * 80)
    print()

    # Track completion status for all benchmarks
    completed = [False] * len(processes)
    returncodes = [None] * len(processes)
    start_times = [time.time()] * len(processes)
    last_status_time = time.time()
    status_interval = config['benchmark']['status_interval']

    all_success = True

    # Monitor all benchmarks in parallel
    while not all(completed):
        current_time = time.time()

        # Check status of all benchmarks
        for i, (process, instance) in enumerate(zip(processes, instances)):
            if completed[i]:
                continue

            returncode = process.poll()
            if returncode is not None:
                # Process has finished
                completed[i] = True
                returncodes[i] = returncode
                elapsed = current_time - start_times[i]
                status = "SUCCESS" if returncode == 0 else "FAILED"
                print(f"✓ Benchmark {i+1}/{len(processes)} ({instance.name}) completed: {status} (elapsed: {elapsed:.1f}s)")

                if returncode != 0:
                    all_success = False
                    print(f"  ERROR: Exit code {returncode}")
                    print(f"   Log file: {log_files[i]}")
                    print("   Last 30 lines of log:")
                    print_log_tail(log_files[i], 30)
                print()

        # Show periodic status updates for all running benchmarks
        if current_time - last_status_time >= status_interval:
            print("=" * 80)
            print(f"Status Update (elapsed: {current_time - min(start_times):.0f}s)")
            print("=" * 80)

            for i, (process, instance) in enumerate(zip(processes, instances)):
                if completed[i]:
                    elapsed = current_time - start_times[i]
                    status = "✓ COMPLETED" if returncodes[i] == 0 else "✗ FAILED"
                    print(f"  [{i+1}/{len(processes)}] {instance.name}: {status} (took {elapsed:.1f}s)")
                else:
                    elapsed = current_time - start_times[i]
                    print(f"  [{i+1}/{len(processes)}] {instance.name}: Running... (elapsed: {elapsed:.0f}s)")
                    print("      Last log lines:")
                    print_log_tail(log_files[i], 3)

            print("=" * 80)
            print()
            last_status_time = current_time

        time.sleep(2)  # Check every 2 seconds

    print("=" * 80)
    print("ALL BENCHMARKS COMPLETED SUCCESSFULLY" if all_success else "SOME BENCHMARKS FAILED - Check logs above")
    print("=" * 80)
    print()

    return all_success


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Multi-LLM Benchmark Orchestrator")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to YAML config file")
    parser.add_argument("--trace", type=str, default="battle")
    args = parser.parse_args()
    config_path = args.config

    # Load instance configurations
    config = load_config(config_path)
    instances = [Instance(**inst) for inst in config['instances']]

    # Calculate and format benchmark durations
    original_duration_str, actual_trace_time_str = calculate_benchmark_durations(config['benchmark'])
    print(f"Run benchmark for duration: {original_duration_str} (original) and {actual_trace_time_str} (actual)")


    # gpu_id
    gpu_str = ""
    if 'gpu_id' in config['benchmark']:
        gpu_id = config['benchmark']['gpu_id']
        print(f"Run benchmark on GPU: {gpu_id} (specified in config)")
        gpu_str = f"gpu_{gpu_id}"
    else:
        gpu_id = None


    gpu_ids = []
    for instance in instances:
        if instance.gpu_id is not None:
            gpu_ids.append(instance.gpu_id)

    number_of_gpus = len(set(gpu_ids))
    if number_of_gpus > 0:
        print(f"Run benchmark on {number_of_gpus} GPUs")
        gpu_str = f"gpu_{gpu_ids[0]}-{gpu_ids[-1]}"
    
    tp_size = config['benchmark']['tp'] if 'tp' in config['benchmark'] else 1

    # Create log directories
    log_base = Path("./logs")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_name = args.config.split("/")[-1].split(".")[0]
    
    log_dir = log_base / config_name / f"run_tp_{tp_size}_{gpu_str}_{len(instances)}_llms_{timestamp}_trace_{original_duration_str}_actual_{actual_trace_time_str}"
    server_log_dir, benchmark_log_dir = create_log_directories(log_dir)

    # Initialize process manager
    process_manager = ProcessManager()

    status = "FAILED"

    print(f"\nRun started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log directory: {log_dir.absolute()}")
    print(f"Number of instances: {len(instances)}")
    print()

    try:
        # Launch servers
        server_log_files = launch_servers(config, instances, server_log_dir, process_manager)

        # Run benchmarks
        all_success = launch_benchmarks(config, args.trace, instances, benchmark_log_dir, process_manager)

        # Summary
        elapsed = time.time() - start_time
        print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")
        if all_success:
            status = "SUCCESS"

    except KeyboardInterrupt:
        print("\n\n  Benchmark interrupted by user (Ctrl+C)")
        process_manager.cleanup_all()
    except Exception as e:
        print(f"\n\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        process_manager.cleanup_all()
    finally:
        # Final cleanup
        process_manager.cleanup_all()
        if status == "SUCCESS":
            # write a file to the log directory to indicate success
            (log_dir / "success").touch()
        else:
            # write a file to the log directory to indicate failure
            (log_dir / "failure").touch()
        print(f"\nDone. Status: {status}")

if __name__ == "__main__":
    main()
