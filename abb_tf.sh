set -ex

# centralized time budget (minutes)
timebudget=600


# abb 1
cp -r ablation/tf/original testharness/tf_cpu
bash build_docker.sh
python3 -u run.py --dll tf --version 2.16 --mode fuzz --check_valid
python3 -u run.py --dll tf --version 2.16 --mode fuzz --time_budget 600 --num_parallel 50 --slurm
python3 -u run.py --dll tf --version 2.16 --mode cov --time_budget 600 --num_parallel 50 --itv 600 --slurm
mv _fuzz_result _fuzz_result_original
mv _cov_result _cov_result_original

# abb2
rm -rf testharness/tf_cpu
cp -r ablation/tf/no_helper testharness/tf_cpu
bash build_docker.sh
python3 -u run.py --dll tf --version 2.16 --mode fuzz --check_valid
python3 -u run.py --dll tf --version 2.16 --mode fuzz --time_budget "$timebudget" --num_parallel 50 --slurm
python3 -u run.py --dll tf --version 2.16 --mode cov --time_budget "$timebudget" --num_parallel 50 --slurm
mv _fuzz_result _fuzz_result_no_helper
mv _cov_result _cov_result_no_helper   

# abb3
rm -rf testharness/tf_cpu
cp -r ablation/tf/no_doc testharness/tf_cpu
bash build_docker.sh
python3 -u run.py --dll tf --version 2.16 --mode fuzz --check_valid
python3 -u run.py --dll tf --version 2.16 --mode fuzz --time_budget "$timebudget" --num_parallel 50 --slurm
python3 -u run.py --dll tf --version 2.16 --mode cov --time_budget "$timebudget" --num_parallel 50 --slurm
mv _fuzz_result _fuzz_result_no_doc
mv _cov_result _cov_result_no_doc

## abb4
rm -rf testharness/tf_cpu
python3 -u run.py --dll tf --version 2.16 --mode fuzz --check_valid
cp -r ablation/tf/no_helper_no_doc testharness/tf_cpu
bash build_docker.sh
python3 -u run.py --dll tf --version 2.16 --mode fuzz --time_budget "$timebudget" --num_parallel 50 --slurm
python3 -u run.py --dll tf --version 2.16 --mode cov --time_budget "$timebudget" --num_parallel 50 --slurm
mv _fuzz_result _fuzz_result_no_helper_no_doc
mv _cov_result _cov_result_no_helper_no_doc
