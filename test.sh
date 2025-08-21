# python3 -u run.py --dll tf --version 2.16 --mode fuzz --check_valid 
# python3 -u run.py --dll tf --version 2.16 --mode fuzz --itv 10 --time_budget 60
# python3 -u run.py --dll tf --version 2.16 --mode cov --check_valid 
# python3 -u run.py --dll tf --version 2.16 --mode cov --itv 10 --time_budget 60

# python3 -u run.py --dll tf --version 2.16 --mode cov --itv 60 --time_budget 1200 --num_parallel 100 --vs pathfinder
python3 -u run.py --dll torch --version 2.2 --mode fuzz --check_valid
