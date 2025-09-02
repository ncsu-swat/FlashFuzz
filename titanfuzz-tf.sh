# python3 -u run.py --dll tf --version 2.16 --mode cov --itv 60 --time_budget 600 --num_parallel 50 --vs titanfuzz --slurm
python3 -u run.py --dll tf --version 2.19 --mode fuzz --itv 50 --time_budget 600 --num_parallel 50 --vs titanfuzz --slurm
