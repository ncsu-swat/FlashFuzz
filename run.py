import argparse
import time
from expmanager import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FlashFuzz.")
    parser.add_argument("--dll", type=str, required=True, help="tf or torch")
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version of the library to use, e.g. 2.2",
    )
    parser.add_argument("--mode", type=str, required=True, help="fuzz or cov")
    parser.add_argument(
        "--apis",
        nargs="*",
        type=str,
        default=None,
        help="List of APIs to test, e.g. api1 api2",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=180,
        help="Time budget in seconds, e.g. 1200 (20 minutes)",
    )

    # cov mode
    parser.add_argument(
        "--itv",
        type=int,
        default=None,
        help="Interval for coverage collection, e.g. 60 (seconds)",
    )

    # Resources
    parser.add_argument(
        "--num_parallel",
        type=int,
        default=1,
        help="Number of parallel dockers to run",
    )
    parser.add_argument(
        "--mem",
        type=int,
        default=16,
        help="Memory limit for each docker (in GB)",
    )

    parser.add_argument(
        "--check_valid",
        action="store_true",
        help="Check the validity of generated inputs",
    )

    # TODO: Add `--crash-report`, `--compilation-check`, and `--validation` arguments

    args = parser.parse_args()

    if args.dll not in ["tf", "torch"]:
        raise ValueError("Invalid DLL specified. Use 'tf' or 'torch'.")

    if args.mode not in ["fuzz", "cov"]:
        raise ValueError("Invalid mode specified. Use 'fuzz' or 'cov'.")

    return args


def main():
    start_time = time.time()
    args = parse_args()
    api_list = f"api_list/{args.dll}{args.version}-flashfuzz.txt"
    with open(api_list, "r") as f:
        apis = f.read().splitlines()

    scheduler = Scheduler()

    if args.dll == "tf" :
        if args.mode == "fuzz":
            if args.check_valid:
                exp = Experiment(
                    dll=args.dll,
                    mode=args.mode,
                    ver=args.version,
                    api="all",
                    cpus=args.num_parallel,
                    mem=args.mem,
                    check_valid=True,
                )
                scheduler.add_experiment(exp)
            else:
                for api in apis:
                    exp = Experiment(
                        dll=args.dll,
                        mode=args.mode,
                        ver=args.version,
                        api=api,
                        cpus=args.num_parallel,
                        time_budget=args.time_budget
                    )
                    scheduler.add_experiment(exp)
        
            
    scheduler.run_all()
    end_time = time.time()
    elapsed_time_s = end_time - start_time
    elapsed_time_m = elapsed_time_s / 60
    elapsed_time_h = elapsed_time_m / 60
    elapsed_str = f"Total Running Time : {(elapsed_time_s):.2f} sec = {(elapsed_time_m):.2f} min = {(elapsed_time_h):.2f} hr"
    print(f"\n{elapsed_str}")

if __name__ == "__main__":
    main()
