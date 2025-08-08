#!/usr/bin/env python3

import os
import glob
import shutil
import argparse

def replace_file_content(file_path: str, old_content: str, new_content: str) -> None:
    """
    Replace old_content with new_content in the specified file.
    If old_content is not found, append new_content to the file.
    """
    try:
        with open(file_path, "r") as file:
            content = file.read()

        if old_content in content:
            content = content.replace(old_content, new_content)

        with open(file_path, "w") as file:
            file.write(content)

        print(f"Updated {file_path}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")


def copy_fuzz_utils(time_budget: int = 180):
    """
    Copy fuzz_utils.h and fuzz_utils.cpp to every subdirectory
    that starts with 'torch', overwriting existing files.
    """
    # Source files
    fuzz_sh = "fuzz.sh"
    build_sh = "build.sh"
    build = "BUILD"
    random_seed = "random_seed.py"

    # Find all directories starting with torch
    torch_dirs = [d for d in glob.glob("tf.*") if os.path.isdir(d)]

    if not torch_dirs:
        print("No directories starting with 'tf' found!")
        return False

    # Copy files to each torch directory
    for torch_dir in torch_dirs:
        api_name = os.path.basename(torch_dir)
        print(f"Processing directory: {torch_dir} (API: {api_name})")

        # target_h = os.path.join(torch_dir, "fuzzer_utils.h")
        # target_cpp = os.path.join(torch_dir, "fuzzer_utils.cpp")
        target_fuzz_sh = os.path.join(torch_dir, "fuzz.sh")

        target_build_sh = os.path.join(torch_dir, "build.sh")

        target_build = os.path.join(torch_dir, "BUILD")

        target_random_seed = os.path.join(torch_dir, "random_seed.py")

        # Copy the files (overwriting if they exist)
        try:
            shutil.copy2(fuzz_sh, target_fuzz_sh)
            shutil.copy2(build_sh, target_build_sh)
            shutil.copy2(build, target_build)
            shutil.copy2(random_seed, target_random_seed)
            replace_file_content(target_fuzz_sh, "{api_name}", api_name)
            replace_file_content(target_build_sh, "{api_name}", api_name)
            replace_file_content(target_fuzz_sh, "{time_budget}", str(time_budget))
            print(f"Copied utils to {torch_dir}")
        except Exception as e:
            print(f"Error copying to {torch_dir}: {e}")

    return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Copy fuzz_utils.h and fuzz_utils.cpp to tf.* directories.")
    parser.add_argument(
        "--time_budget",
        type=int,
        default=180,
        help="Time budget in seconds for the fuzzing process, e.g. 180 (3 minutes)",
    )
    args = parser.parse_args()

    print("Copying fuzz_utils.h and fuzz_utils.cpp to tf.* directories...")
    success = copy_fuzz_utils(args.time_budget)
    if success:
        print("Done!")
    else:
        print("Operation failed!")
