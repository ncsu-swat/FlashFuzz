import argparse
import os
import glob
import multiprocessing
from functools import partial

def run_fuzzer_for_seed(seed: str, dir_basename: str):
    """
    Worker function that runs the fuzzer for a single seed file.
    This function is executed by the parallel worker processes.
    """
    # Define paths relative to the current working directory (the seed directory)
    binary_path = "./fuzz"
    profraw_file = f"{seed}.profraw"
    
    # Execute the command to generate the coverage profile for one seed
    os.system(f"LLVM_PROFILE_FILE='{profraw_file}' {binary_path} {seed}")

def main():
    parser = argparse.ArgumentParser(description="Generate coverage files.")
    parser.add_argument("--dll", type=str, required=True, help="Input directory containing coverage data.")
    parser.add_argument("--api", type=str, required=True, help="API name")
    parser.add_argument("--itv", type=int, default=60, help="Interval for coverage collection in seconds.")
    args = parser.parse_args()

    if args.dll == "tf" and args.api != "all":
        os.system("cp /root/tensorflow/bazel-bin/tensorflow/libtensorflow_*.so* .")
        seeds_dir = f"corpus_itv_{args.itv}"
        
        # Get absolute paths for files to be copied
        binary_path = os.path.realpath("./fuzz")
        libtensorflow_path = os.path.realpath("libtensorflow_cc.so.2")
        libtensorflow_framework_path = os.path.realpath("libtensorflow_framework.so.2")

        # List all directories in the seeds directory
        dirs = [d for d in glob.glob(f"{seeds_dir}/*") if os.path.isdir(d)]
        cwd = os.getcwd()

        # Sequentially process each directory
        for d in dirs:
            os.chdir(cwd)

            # Copy the binary and libraries to the target directory
            os.system(f"cp {binary_path} {d}/")
            os.system(f"cp {libtensorflow_path} {d}/")
            os.system(f"cp {libtensorflow_framework_path} {d}/")

            # Change into the target directory for processing
            os.chdir(d)
            
            # Get a list of all files in the directory
            all_files = glob.glob("*")
            # Filter the list to include only the actual input seeds
            # Exclude the copied binary, libraries, subdirectories, and previous run artifacts
            input_seeds = [
                f for f in all_files 
                if f not in ['fuzz', 'libtensorflow_cc.so.2', 'libtensorflow_framework.so.2'] 
                and not os.path.isdir(f) 
                and not f.endswith(('.profraw', '.profdata'))
            ]

            if not input_seeds:
                print(f"No input seeds found in {d}, skipping.")
                continue

            # Use partial to create a version of the worker function 
            # with the directory basename argument already filled in.
            dir_basename = os.path.basename(os.getcwd())
            worker_func = partial(run_fuzzer_for_seed, dir_basename=dir_basename)

            # This is the parallel part:
            # Create a pool of 16 worker processes to run the fuzzer on all seeds
            print(f"Running fuzzer in parallel on {len(input_seeds)} seeds in {d}...")
            with multiprocessing.Pool(processes=16) as pool:
                pool.map(worker_func, input_seeds)
            print(f"Fuzzing complete for {d}.")

            # This step runs only after all parallel fuzzing tasks are finished
            print(f"Merging profraw files in {d}...")
            os.system(f"llvm-profdata merge -sparse -o merged.profdata *.profraw")
            print(f"Merging complete for {d}.")

            # delete everything except the merged.profdata file
            for f in all_files:
                if f != "merged.profdata":
                    os.remove(f)
        # Return to the original directory after processing all directories
        os.chdir(cwd)

if __name__ == "__main__":
    # This check is important for robust multiprocessing on all platforms
    main()
