# Novita benchmark

This directory contains the benchmark scripts for the Novita trace

## Install kvcached

Please refer to the [kvcached README](../../README.md) for more detailed installation instructions.

### Option 1: Install from PyPI

```bash
pip install kvcached --no-build-isolation
```

Set the environmental variables to enable kvcached.

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export KVCACHED_IPC_NAME=SGLANG
```

### Option 2: Run kvcached with Docker

Pull the docker image for kvcached with SGLang engine.

```bash
docker pull ghcr.io/ovg-project/kvcached-sglang:latest   # kvcached-v0.1.1-sglang-v0.5.3
```

Run the docker container.

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --env "HF_TOKEN=<secret>" \
  -v /dev/shm:/shm \
  -v /.cache:/root/.cache \
  -v /data/novita-trace:/data/novita-trace \
  --ipc=host \
  --network=host \
  --privileged \
  --name kvcached-sglang \
  ghcr.io/ovg-project/kvcached-sglang:latest \
  bash
```

Attach to the container.

```bash
docker exec -it kvcached-sglang bash
```

### Test the installation

```bash
# for sglang
python -m sglang.launch_server --model meta-llama/Llama-3.2-1B --disable-radix-cache --port 30000
python -m sglang.bench_serving --backend sglang-oai --model meta-llama/Llama-3.2-1B --dataset-name sharegpt --request-rate 10 --num-prompts 1000 --port 30000
```

## Prepare the benchmark

### Prepare the trace

First, we can save the trace to be replayed to a directory that can be mounted to the docker (e.g. `data/novita-trace`).

The directory includes multiple csv files, each corresponding to a LLM and including the requests that are sent to the LLM.

The csv file should include the following columns:
- "Timestamp": the timestamp of the request, e.g. `2025-04-06 00:02:15.673000`
- "Prompt Len": the length of the prompt, e.g. `2048`
- "Output Len": the length of the output, e.g. `256`

### Download the example benchmarking scripts

Download the example benchmarking scripts.

```bash 
git clone -b novita-bench https://github.com/shanyu-sys/kvcached-open.git
cd kvcached-open/benchmarks/bench_novita
```

### Adjust the configuration

Adjust the configuration in the `config.yml` file to match your trace file and model choices.

All colocated model instances must fit within the memory of a single GPU. The memory footprint of each model instance can be approximated as `model_weights_size + captured CUDA graph size + runtime memory`.

The captured CUDA graph could add around 4 GB, though the exact size may vary depending on the model. 

The runtime memory consists of the dynamically allocated KV-cache as well as temporary buffers used during the model’s forward pass.


### Run the benchmark

```bash
python bench_multi_llm.py
```

The benchmark will run multiple colocated model instances with the SGLang engine, and send the corresponding requests (read from the trace file) to the model instances.

All the results will be saved in the `logs` directory, including the server logs and the benchmark logs.


### Metrics

I’m continuing to work on the metrics analysis. The current plan is to include the following metrics:

- **Revenue over time**: A plot showing how revenue changes over time. Revenue is computed as `token throughput × price per token of the model`, aggregated across all model instances. Compare the revenue of Prism and the baseline.

- **Average utilization boost**: This measures how much more workload Prism can handle compared to the baseline for the same hardware budget. For example, if Prism can serve 4 models using the same hardware where the baseline can only serve 2, then the utilization boost is the ratio of Prism’s total token throughput to the baseline’s.

- **Cost savings**: This shows how much fewer GPUs Prism requires to serve the same number of models. It can be calculated as: `cost savings = (GPUs required by Prism) / (GPUs required by the baseline)`.