import argparse
import os
import subprocess
import glob
from typing import Optional


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


def build_tf() -> None:
    tf_dirs = glob.glob("tf.*")
    total_dirs = len(tf_dirs)
    print(f"Found {total_dirs} TensorFlow API directories to build.")
    success_count = 0
    for tf_dir in tf_dirs:
        api_name = os.path.basename(tf_dir)
        print(f"Building TensorFlow API: {api_name}")

        cmd = "bash build.sh"
        ret_code, output = run_command(cmd, tf_dir)

        if ret_code != 0:
            print(f"Error building {api_name}: {output}")
        else:
            print(f"Successfully built {api_name}")
            success_count += 1

    print(f"Built {success_count}/{total_dirs} TensorFlow APIs successfully.")


def build_tf_cov() -> None:
    print("Building TensorFlow coverage harness...")
    os.system("cp -r template/tf_cpu_cov/* .")
    os.system("cp template/tf_cpu_cov/copy.py .")
    os.system("python3 copy.py")
    build_tf()


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
    args = parser.parse_args()

    if args.dll == "tf" and args.mode == "cov":
        build_tf_cov()
    elif args.dll == "tf" and args.mode == "fuzz":
        build_tf()
    # Add logic for torch if necessary


if __name__ == "__main__":
    main()
