# Multi-LLM benchmark

This directory contains the benchmark scripts for the Multi-LLM serving 

## Install kvcached

Please refer to the [kvcached README](../../README.md) for more detailed installation instructions.

You can run the entire benchmark using the kvcached SGLang docker image.


## Prepare the benchmark

### Download the example benchmarking scripts

Download the example benchmarking scripts.

```bash 
cd ~
git clone -b multi-llm-test https://github.com/shanyu-sys/kvcached-open.git
cd kvcached-open/benchmarks/multi-llm-testing
```

### Adjust the configuration

Adjust the configuration in the `config_prism_gpu_{i}.yml` or `config_baseline_8_gpu` file if needed. 
For example, 
* Run shorter trace duration by adjusting the end timestamp (e.g. with start timestamp "2025-08-07 00:00:00.000000" and end timestamp "2025-08-07 00:05:00.000000" will send requests in the first 5 minutes). For your test run, please make sure to change the end timestamp to a shorter duration (e.g. "2025-08-07 00:05:00.000000") in all the configration yaml files. Default trace duration is 1 day.
* Adjust the gpu_id if it is occupied by other processes.
* Adjust the port if it is occupied by other processes. Should make sure that all the ports are different (in all config files if run on multiple gpus).


### Run the benchmark


#### Prism test

Run prism test on gpu 0
```bash
python bench_multi_llm.py --config config_prism_gpu_0.yml
```

Run prism test on gpu 1
```bash
python bench_multi_llm.py --config config_prism_gpu_1.yml
```

Run prism test on gpu 2
```bash
python bench_multi_llm.py --config config_prism_gpu_2.yml
```

Run prism test on gpu 3
```bash
python bench_multi_llm.py --config config_prism_gpu_3.yml
```

All server and benchmark logs will be saved in the `logs/config_prism_gpu_{i}` directory. Each run will generate a new directory with the timestamp.


#### Baseline test

Run baseline test on 0-7 gpus
```bash
python bench_multi_llm.py --config config_baseline_8_gpu.yml 
```

All server and benchmark logs will be saved in the `logs/config_baseline_8_gpu` directory.
