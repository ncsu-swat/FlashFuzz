# FlashFuzz
A framework to employ coverage-guided fuzzing to test Deep Learning APIs at scale.

# Structure
```
- testharness: Contains the test harness for fuzzing.
```
# 1.Setup

```bash
bash build_docker.sh
```

# 2. Basic Usage
- Common Flags
    - `--dll`: Target DL library, Should be obne of `tf`, `torch`
    - `--version`: Version of the DL library, currently supported versions are `2.16` and `2.19`(fuzz only) for tensorflow.
    - `--mode`: Should be one of `fuzz`, `cov`
    - `--num_parallel`: Number of parallel experiments to run.

## 2.1 Fuzzing
```bash
python3 -u run.py --dll tf --version 2.16 --mode fuzz 
```
Results are stored in `_fuzz_result/` directory.

## 2.1.1 Check the validity of the test harnesses
```bash
python3 -u run.py --dll tf --version 2.16 --mode fuzz --check_valid 
```
Results are stored in `_fuzz_result/build_status` directory.

## 2.1.2 Adjust the time budget for fuzzing
```bash
python3 -u run.py --dll tf --version 2.16 --mode fuzz --time_budget 300
```
This will set the time budget for each fuzzing run to 300 seconds. The default is 180 seconds.

## 2.2 Coverage
```bash
python3 -u run.py --dll tf --version 2.16 --mode cov
```
Results are stored in `_cov_result/` directory.
