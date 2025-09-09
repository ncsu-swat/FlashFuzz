set -ex

# # abb 1
cp -r ablation/torch/original testharness/torch_cpu
bash build_docker.sh
python3 -u run.py --dll torch --version 2.2 --mode fuzz --check_valid 
python3 -u run.py --dll torch --version 2.2 --mode fuzz --time_budget 600 --num_parallel 50 --slurm 
python3 -u run.py --dll torch --version 2.2 --mode cov --time_budget 600 --num_parallel 50 --slurm 
mv _fuzz_result _fuzz_result_original
mv _cov_result _cov_result_original

# abb2
rm -rf testharness/torch_cpu
cp -r ablation/torch/no_helper testharness/torch_cpu
bash build_docker.sh
python3 -u run.py --dll torch --version 2.2 --mode fuzz --check_valid 
python3 -u run.py --dll torch --version 2.2 --mode fuzz --time_budget 600 --num_parallel 50 --slurm
python3 -u run.py --dll torch --version 2.2 --mode cov --time_budget 600 --num_parallel 50 --slurm
mv _fuzz_result _fuzz_result_no_helper
mv _cov_result _cov_result_no_helper   

# # abb3
rm -rf testharness/torch_cpu
cp -r ablation/torch/no_doc testharness/torch_cpu
bash build_docker.sh
python3 -u run.py --dll torch --version 2.2 --mode fuzz --check_valid 
python3 -u run.py --dll torch --version 2.2 --mode fuzz --time_budget 600 --num_parallel 50 --slurm
python3 -u run.py --dll torch --version 2.2 --mode cov --time_budget 600 --num_parallel 50 --slurm
mv _fuzz_result _fuzz_result_no_doc
mv _cov_result _cov_result_no_doc

## abb4
rm -rf testharness/torch_cpu
python3 -u run.py --dll torch --version 2.2 --mode fuzz --check_valid 
cp -r ablation/torch/no_helper_no_doc testharness/torch_cpu
bash build_docker.sh
python3 -u run.py --dll torch --version 2.2 --mode fuzz --time_budget 600 --num_parallel 50 --slurm
python3 -u run.py --dll torch --version 2.2 --mode cov --time_budget 600 --num_parallel 50 --slurm
mv _fuzz_result _fuzz_result_no_helper_no_doc
mv _cov_result _cov_result_no_helper_no_doc
