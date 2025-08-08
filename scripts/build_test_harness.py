import argparse
import os
import subprocess
import glob
from typing import Optional
import concurrent.futures


def run_command(cmd: str, cwd: str, timeout: Optional[float] = None) -> tuple[int, str]:
    """Run a command with timeout and return result"""
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
        )

        try:
            stdout, _ = process.communicate(timeout=timeout)
            return process.returncode, stdout
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "TIMEOUT: Command exceeded time limit"
    except Exception as e:
        return -2, f"ERROR: {str(e)}"


def build_one_tf_api(tf_dir: str) -> tuple[str, int, str]:
    """Build a single TensorFlow API and return the result."""
    api_name = os.path.basename(tf_dir)
    print(f"Building TensorFlow API: {api_name}")
    cmd = "bash build.sh > build.log"
    ret_code, output = run_command(cmd, tf_dir)
    return api_name, ret_code, output


def build_tf() -> None:
    """Find and build all TensorFlow APIs in parallel."""
    tf_dirs = glob.glob("tf.*")
    total_dirs = len(tf_dirs)
    if total_dirs == 0:
        print("No TensorFlow API directories found to build.")
        return

    print(f"Found {total_dirs} TensorFlow API directories to build in parallel.")
    success_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(build_one_tf_api, tf_dir) for tf_dir in tf_dirs]

        for future in concurrent.futures.as_completed(futures):
            try:
                api_name, ret_code, output = future.result()
                if ret_code != 0:
                    print(f"Error building {api_name}: {output}")
                else:
                    print(f"Successfully built {api_name}")
                    success_count += 1
            except Exception as e:
                print(f"An exception occurred during build: {e}")

    print(f"Built {success_count}/{total_dirs} TensorFlow APIs successfully.")


def check_tf_build() -> None:
    print("Checking TensorFlow build status...")
    tf_dirs = glob.glob("tf.*")
    if not tf_dirs:
        print("No TensorFlow API directories found.")
        return

    build_success: int = 0
    success_build_apis: list[str] = []
    fail_build_apis: list[list[str]] = [] 
    total: int = len(tf_dirs)
    for tf_dir in tf_dirs:
        api_name = os.path.basename(tf_dir)
        keyword = api_name.split(".")[-1]
        binary = f"{tf_dir}/fuzz"
        source_file = f"{tf_dir}/fuzz.cpp"
        status = "OK"
        with open(source_file, "r") as f:
            content = f.read()
            # Check if the keyword is in the source file, ignore case sensitivity
            if keyword.lower() in content.lower():
               print(f"Keyword '{keyword}' found in {source_file}.")
            else:
               print(f"Keyword '{keyword}' NOT found in {source_file}.")
               status = "TESTHARNESS FAIL"

        if os.path.exists(binary):
            pass
        else:
            status = "BUILD FAIL"
        if status == "OK":
            build_success += 1
            success_build_apis.append(api_name)
        else:
            fail_build_apis.append([api_name, status])

    print(f"Build status: {build_success}/{total} TensorFlow APIs built successfully.")
    with open("success_apis.txt", "w") as f:
        f.write("\n".join(success_build_apis))
    with open("fail_apis.txt", "w") as f:
        for api_name, status in fail_build_apis:
            f.write(f"{api_name}: {status}\n")
    with open("build_summary.txt", "w") as f:
        f.write(f"Build status: {build_success}/{total} TensorFlow APIs built successfully.\n")


def build_tf_fuzz(time_budget: int=180, no_compile: bool=False) -> None:
    print("Building TensorFlow fuzz harness...")
    os.system("cp -r template/tf_cpu/* .") # Copy all files from the template directory
    os.system(f"python3 -u copy.py --time_budget {time_budget}") # Copy the fuzz.sh and build.sh files with proper settings
    if not no_compile:
        build_tf()
    check_tf_build()


def build_tf_cov(time_budget: int=180, no_compile: bool=False) -> None:
    print("Building TensorFlow coverage harness...")
    os.system("cp -r template/tf_cpu_cov/* .") # Copy all files from the template directory
    os.system(f"python3 -u copy.py --time_budget {time_budget}") # Copy the fuzz.sh and build.sh files with proper settings
    if not no_compile:
        build_tf()
    check_tf_build()


def main():
    parser = argparse.ArgumentParser(
        description="Build the test harness for FlashFuzz."
    )
    parser.add_argument(
        "--dll",
        type=str,
        required=True,
        choices=["tf", "torch"],
        help="The deep learning library to use: 'tf' or 'torch'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["fuzz", "cov"],
        help="The mode to run in: 'fuzz' or 'cov'.",
    )
    parser.add_argument(
        "--check_build",
        action="store_true",
        help="Check if the build was successful.",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        required=False,
        help="Time budget for the experiment in seconds.",
        default=180,
    )
    parser.add_argument(
        "--no-compile",
        required=False,
        action="store_true",
        help="Skip the compilation step. This is useful for testing configurations (e.g. time budget).",
    )
    args = parser.parse_args()

    if args.dll == "tf" and args.mode == "cov":
        if args.check_build:
            check_tf_build()
        else:
            build_tf_cov(no_compile=args.no_compile, time_budget=args.time_budget)
    elif args.dll == "tf" and args.mode == "fuzz":
        if args.check_build:
            check_tf_build()
        else:
            build_tf_fuzz(no_compile=args.no_compile, time_budget=args.time_budget)

    # Add logic for torch if necessary


if __name__ == "__main__":
    main()
