import argparse
import os
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
        default=None,
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

    args = parser.parse_args()

    if args.dll not in ["tf", "torch"]:
        raise ValueError("Invalid DLL specified. Use 'tf' or 'torch'.")

    if args.mode not in ["fuzz", "cov"]:
        raise ValueError("Invalid mode specified. Use 'fuzz' or 'cov'.")

    return args


def main():
    start_time = time.time()

    args = parse_args()
