import enum
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import glob
import shutil
import time
from typing import Callable, Optional
import re

class Status(enum.Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# while loop until the user presses Ctrl+C

def loop_until_ctrl_c(callback: Optional[Callable[[], None]] = None, interval: float = 1.0):
    """
    Run repeatedly until the user presses Ctrl+C.
    If a callback is provided, it will be invoked each iteration.
    """
    try:
        while True:
            if callback:
                try:
                    callback()
                except Exception:
                    # Ignore callback errors to keep the loop running
                    pass
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting loop.")


class Experiment():
    def __init__(self, dll: str, mode: str, ver: str, api: str, cpus: int = 16, mem: int = 16, check_valid: bool = False, time_budget: int = 180, itv: int = 60):
        self.dll = dll
        self.mode = mode
        self.ver = ver
        self.api = api
        self.cpus = cpus
        self.mem = mem
        self.check_valid = check_valid
        self.time_budget = time_budget
        self.status = Status.NOT_STARTED
        self.image_name = f"ncsuswat/flashfuzz:{self.dll}{self.ver}-{self.mode}"
        self.container_name = f"{self.api}_{self.dll}{self.ver}_{self.mode}"
        self.container_id = None
        self.itv = itv
        if self.api == "all":
            self.result_dir = f"./_{self.mode}_result/{self.dll}{self.ver}-{self.mode}-{self.time_budget}s/"
        else:
            self.result_dir = f"./_{self.mode}_result/{self.dll}{self.ver}-{self.mode}-{self.time_budget}s/{self.api}/"
        self.check_image()

    def check_image(self):
        cmd = f"docker images -q {self.image_name}"
        proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        out, _ = proc.communicate()
        image_available = out.decode().strip() != ""

        if not image_available:
            print(f"Image {self.image_name} not found. Please build it first.")
            raise RuntimeError(f"Image {self.image_name} not found. Please build it first.")
        # print(f"Image {self.image_name} is available.")

    def classify_with_itv(self, itv: int):
        # check if dir exists
        if not os.path.exists(self.result_dir):
            print(f"Result directory {self.result_dir} does not exist. Cannot classify.")
            return

        # Allow defaulting to instance interval
        itv = itv or getattr(self, "itv", 60)
        if itv <= 0:
            print("Interval must be a positive integer (seconds).")
            return

        # Determine candidate experiment output dirs
        dirs = glob.glob(f"{self.result_dir}/{self.dll}.*")
        if not dirs:
            print(f"No directories found in {self.result_dir} matching pattern {self.dll}.*")
            return

        for base_dir in dirs:
            seed_dir = os.path.join(base_dir, "corpus")
            if not os.path.isdir(seed_dir):
                print(f"Corpus directory not found: {seed_dir}")
                return

            # Collect all files (recursively) in corpus
            files: list[str] = []
            for root, _dirs, fnames in os.walk(seed_dir):
                for fname in fnames:
                    fpath = os.path.join(root, fname)
                    if os.path.isfile(fpath):
                        files.append(fpath)

            if not files:
                print(f"No files found in corpus: {seed_dir}")
                return

            # Sort by modification time (approx creation order on Linux)
            files.sort(key=lambda p: os.path.getmtime(p))
            t0 = os.path.getmtime(files[0])

            # Destination base for buckets
            dest_base = os.path.join(base_dir, f"corpus_itv_{itv}")
            os.makedirs(dest_base, exist_ok=True)

            bucket_counts: dict[str, int] = {}

            for f in files:
                dt = os.path.getmtime(f) - t0
                bucket = int(dt // itv)
                start = bucket * itv
                end = (bucket + 1) * itv
                bucket_name = f"{start}-{end}"
                bucket_dir = os.path.join(dest_base, bucket_name)
                os.makedirs(bucket_dir, exist_ok=True)

                # Avoid overwriting files with same basename
                base_name = os.path.basename(f)
                target = os.path.join(bucket_dir, base_name)
                if os.path.exists(target):
                    name, ext = os.path.splitext(base_name)
                    i = 1
                    while os.path.exists(target):
                        target = os.path.join(bucket_dir, f"{name}_{i}{ext}")
                        i += 1
                try:
                    shutil.copy2(f, target)
                except Exception as e:
                    print(f"Failed to copy {f} -> {target}: {e}")
                    continue

                bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1

            # Print summary for this base_dir
            total = sum(bucket_counts.values())
            print(f"Classified {total} files from {seed_dir} into {len(bucket_counts)} buckets under {dest_base}.")

    def create_docker_container(self):
        cmd = f'docker create --name {self.container_name} {self.image_name}'
        try:
            proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self.container_id = proc.stdout.strip()
            print(f"Created container {self.container_name} with ID: {self.container_id}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create container {self.container_name}.")
            print(f"Stderr: {e.stderr}")
            raise

    def start_docker_container(self):
        # cmd = f"docker run -itd --cpus {self.cpus} -m {self.mem}g --name {self.container_name} {self.image_name}"
        cmd = f"docker run -itd --cpus {self.cpus} --name {self.container_name} {self.image_name}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Started container {self.container_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start container {self.container_name}.")
            print(f"Stderr: {e.stderr}")
            raise

    def execute_command(self, command: str):
        cmd = f'docker exec {self.container_name} sh -c "{command}"'
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        except Exception:
            pass

    def stop_docker_container(self):
        cmd = f"docker stop {self.container_name}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Stopped container {self.container_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop container {self.container_name}.")
            print(f"Stderr: {e.stderr}")
            raise

    def remove_docker_container(self, prune_volumes: bool = True):
        # Use -v to remove anonymous volumes associated with the container
        vol_flag = "-v" if prune_volumes else ""
        cmd = f"docker rm {vol_flag} {self.container_name}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            if prune_volumes:
                print(f"Removed container {self.container_name} and its anonymous volumes.")
            else:
                print(f"Removed container {self.container_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove container {self.container_name}.")
            print(f"Stderr: {e.stderr}")
            raise

    def copy_results_from_container(self, src: str, dest: str):
        cmd = f"docker cp {self.container_name}:{src} {dest}"
        subprocess.run(f"mkdir -p {self.result_dir}", shell=True, check=True)
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Copied results from {src} to {dest}.")
        except Exception:
            pass
    
    def copy_files_to_container(self, src: str, dest: str):
        cmd = f"docker cp {src} {self.container_name}:{dest}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Copied files from {src} to {dest} in container {self.container_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to copy files to container {self.container_name}.")
            print(f"Stderr: {e.stderr}")
            raise

    def tf_fuzz(self):
        self.status = Status.RUNNING
        try:
            self.check_image()
            self.start_docker_container()
            self.execute_command(f"cd /root/tensorflow/fuzz/ && python3 build_test_harness.py --dll {self.dll} --mode {self.mode} --ver {self.ver} --time_budget {self.time_budget} --no-compile")
            self.execute_command(f"cd /root/tensorflow/fuzz/{self.api} && bash fuzz.sh > execution.log")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}/execution.log", self.result_dir)
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}/fuzz-0.log", self.result_dir)
            # self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}/corpus", self.result_dir)
            self.status = Status.COMPLETED
        except Exception:
            self.status = Status.FAILED

    def tf_check_valid(self):
        self.status = Status.RUNNING
        try:
            self.check_image()
            self.start_docker_container()
            self.execute_command(f"cd /root/tensorflow/fuzz/ && python3 -u build_test_harness.py --dll {self.dll} --mode {self.mode} --check_build > check.log")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}", f"{self.result_dir}/{self.api}")
            os.makedirs(f"{self.result_dir}/build_status", exist_ok=True)
            self.copy_results_from_container(f"/root/tensorflow/fuzz/success_apis.txt", f"{self.result_dir}/build_status/")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/fail_apis.txt", f"{self.result_dir}/build_status/")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/build_summary.txt", f"{self.result_dir}/build_status/")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/check.log", f"{self.result_dir}/build_status/")
            with open(f"{self.result_dir}/build_status/build_summary.txt", "r") as f:
                summary = f.read()
                print(f"Build Summary: {summary}")
            self.status = Status.COMPLETED
        except Exception:
            self.status = Status.FAILED

    def tf_cov_api(self):
        self.status = Status.RUNNING
        try:
            self.check_image()
            self.start_docker_container()
            self.execute_command(f"cd /root/tensorflow/fuzz/{self.api}  && python3 coverage_fuzzing.py --interval {self.itv} --max-time {self.time_budget} --api {self.api}")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}/coverage_data", f"{self.result_dir}/coverage_data")
            self.status = Status.COMPLETED
        except Exception:
            self.status = Status.FAILED

    # --- torch support ---
    def torch_fuzz(self):
        self.status = Status.RUNNING
        try:
            self.check_image()
            self.start_docker_container()
            self.execute_command(f"cd /root/fuzz/ && python3 build_test_harness.py --dll {self.dll} --mode {self.mode} --ver {self.ver} --time_budget {self.time_budget} --no-compile")
            self.execute_command(f"cd /root/fuzz/{self.api} && bash fuzz.sh > execution.log")
            self.copy_results_from_container(f"/root/fuzz/{self.api}/execution.log", self.result_dir)
            self.copy_results_from_container(f"/root/fuzz/{self.api}/fuzz-0.log", self.result_dir)
            # self.copy_results_from_container(f"/root/fuzz/{self.api}/corpus", self.result_dir)
            self.status = Status.COMPLETED
        except Exception:
            self.status = Status.FAILED

    def torch_check_valid(self):
        self.status = Status.RUNNING
        try:
            self.check_image()
            self.start_docker_container()
            self.execute_command(f"cd /root/fuzz/ && python3 -u build_test_harness.py --dll {self.dll} --mode {self.mode} --check_build > check.log")
            self.copy_results_from_container(f"/root/fuzz/{self.api}", f"{self.result_dir}/{self.api}")
            os.makedirs(f"{self.result_dir}/build_status", exist_ok=True)
            self.copy_results_from_container(f"/root/fuzz/success_apis.txt", f"{self.result_dir}/build_status/")
            self.copy_results_from_container(f"/root/fuzz/fail_apis.txt", f"{self.result_dir}/build_status/")
            self.copy_results_from_container(f"/root/fuzz/build_summary.txt", f"{self.result_dir}/build_status/")
            self.copy_results_from_container(f"/root/fuzz/check.log", f"{self.result_dir}/build_status/")
            with open(f"{self.result_dir}/build_status/build_summary.txt", "r") as f:
                summary = f.read()
                print(f"Build Summary: {summary}")
            self.status = Status.COMPLETED
        except Exception:
            self.status = Status.FAILED

    def torch_cov_api(self):
        self.status = Status.RUNNING
        try:
            self.check_image()
            self.start_docker_container()
            self.execute_command(f"cd /root/fuzz/{self.api}  && python3 coverage_fuzzing.py --interval {self.itv} --max-time {self.time_budget} --api {self.api}")
            self.copy_results_from_container(f"/root/fuzz/{self.api}/coverage_data", f"{self.result_dir}/coverage_data")
            self.status = Status.COMPLETED
        except Exception:
            self.status = Status.FAILED

    def merge_coverage_files(self) -> None:
        print("Merging coverage files...")
        output_dir = f"{self.result_dir}/all"
        os.makedirs(output_dir, exist_ok=True)
        previous_interval_name: Optional[str] = None
        for idx in range(0, self.time_budget, self.itv):
            interval_start = idx
            interval_end = min(idx + self.itv, self.time_budget)
            interval_name = f"{interval_start}-{interval_end}"
            interval_dir = os.path.join(output_dir, interval_name)
            os.makedirs(interval_dir, exist_ok=True)

            api_dirs = glob.glob(f"{self.result_dir}/{self.dll}.*")
            if not api_dirs:
                print(f"No {self.dll} API directories found.")
                return
            for api_dir in api_dirs:
                profraw_file = os.path.join(api_dir, "coverage_data", f"{interval_name}", "*.profraw")
                profraw_files = glob.glob(profraw_file)
                if not profraw_files:
                    print(f"No profraw files found for {api_dir} in interval {interval_name}.")
                    continue
                for profraw in profraw_files:
                    shutil.copy2(profraw, interval_dir)
                    print(f"Copied {profraw} to {interval_dir}")
            # Merge inside container; choose base_root by dll
            base_root = "/root/tensorflow/fuzz" if self.dll == "tf" else "/root/fuzz"
            try:
                self.check_image()
                self.start_docker_container()
                self.copy_files_to_container(f"{output_dir}/{interval_name}", f"{base_root}/")
                if previous_interval_name:
                    self.copy_files_to_container(f"{output_dir}/{previous_interval_name}", f"{base_root}/{previous_interval_name}")
                if not previous_interval_name:
                    self.execute_command(f"cd {base_root} && python3 merge_profraw.py --dll {self.dll} --dir {interval_name}")
                else:
                    self.execute_command(f"cd {base_root} && python3 merge_profraw.py --dll {self.dll} --dir {interval_name} --previous {previous_interval_name}/merged.profdata")
                self.copy_results_from_container(f"{base_root}/{interval_name}/merged.profdata", f"{interval_dir}/merged.profdata")
                previous_interval_name = interval_name
                # delete the profraw files to save space
                profraw_files = glob.glob(os.path.join(interval_dir, "*.profraw"))
                for profraw_file in profraw_files:
                    os.remove(profraw_file)
            except KeyboardInterrupt:
                print(f"\nKeyboard interrupt received.")
            except Exception as e:
                print(f"Failed to merge profraw files for interval {interval_name}: {e}")
            finally:
                try:
                    self.stop_docker_container()
                    self.remove_docker_container()
                except Exception:
                    pass

    def get_coverage_results(self):
        print("Getting coverage results...")
        output_dir = f"{self.result_dir}/all"
        os.makedirs(output_dir, exist_ok=True)
        for idx in range(0, self.time_budget, self.itv):
            interval_start = idx
            interval_end = min(idx + self.itv, self.time_budget)
            interval_name = f"{interval_start}-{interval_end}"
            interval_dir = os.path.join(output_dir, interval_name)
            os.makedirs(interval_dir, exist_ok=True)

            # copy the merged.profdata file to the output directory
            merged_profdata_file = os.path.join(interval_dir, "merged.profdata")
            if not os.path.exists(merged_profdata_file):
                print(f"No merged.profdata file found for interval {interval_name}.")
                continue

            # Run results extraction inside container; choose base_root by dll
            base_root = "/root/tensorflow/fuzz" if self.dll == "tf" else "/root/fuzz"
            try:
                self.check_image()
                self.start_docker_container()
                self.copy_files_to_container(merged_profdata_file, f"{base_root}/")
                if self.dll == "tf":
                    self.execute_command(f"cd {base_root} && python3 get_coverage_results.py --binary /root/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so.2.16.1 --require tensorflow/core/kernels --coverage_file merged.profdata --out {interval_name}.txt")
                else:
                    self.execute_command(f"cd {base_root} && python3 get_coverage_results.py --dll torch --coverage_file merged.profdata --out {interval_name}.txt")
                self.copy_results_from_container(f"{base_root}/{interval_name}.txt", f"{output_dir}/{interval_name}.txt")
                self.copy_results_from_container(f"{base_root}/coverage_html", f"{interval_dir}/coverage_html")
                # remove merged_profdata_file to save space
                os.remove(merged_profdata_file)
            except KeyboardInterrupt:
                print(f"\nKeyboard interrupt received.")
            except Exception as e:
                print(f"Failed to get coverage results for interval {interval_name}: {e}")
            finally:
                try:
                    self.stop_docker_container()
                    self.remove_docker_container()
                except Exception:
                    pass
        # print the coverage summary
        for idx in range(0, self.time_budget, self.itv):
            interval_start = idx
            interval_end = min(idx + self.itv, self.time_budget)
            interval_name = f"{interval_start}-{interval_end}"
            interval_dir = os.path.join(output_dir, interval_name)
            coverage_results_file = f"{interval_dir}.txt"
            if not os.path.exists(coverage_results_file):
                print(f"No coverage_results.txt file found for interval {interval_name}.")
                continue
            with open(coverage_results_file, "r") as f:
                coverage_summary = f.read()
                pattern = r"(?<=Covered branches: )\d+"
                coverage_number = re.search(pattern, coverage_summary)
                if coverage_number:
                    coverage_number = coverage_number.group()
                    print(f"{interval_name}: {coverage_number}")


    def run(self):
        if self.dll == "tf":
            if self.check_valid:
                self.tf_check_valid()
            elif self.mode == "fuzz":
                self.tf_fuzz()
            elif self.mode == "cov":
                if self.api != "all":
                    self.tf_cov_api()
        elif self.dll == "torch":
            if self.check_valid:
                self.torch_check_valid()
            elif self.mode == "fuzz":
                self.torch_fuzz()
            elif self.mode == "cov":
                if self.api != "all":
                    self.torch_cov_api()

class Scheduler():
    def __init__(self, num_parallel: int = 1):
        self.experiments: list[Experiment] = []
        self.num_parallel = num_parallel

    def add_experiment(self, experiment: Experiment):
        self.experiments.append(experiment)

    def _run_experiment(self, exp: Experiment):
        """Helper function to run a single experiment and handle cleanup."""
        try:
            print(f"Running experiment for {exp.api}...")
            exp.run()
            if exp.status == Status.COMPLETED:
                print(f"Experiment for {exp.api} completed successfully.")
            else:
                print(f"Experiment for {exp.api} failed.")
        except KeyboardInterrupt:
            print(f"\nKeyboard interrupt received. Cleaning up experiment for {exp.api}...")
            exp.status = Status.FAILED
        finally:
            try:
                exp.stop_docker_container()
                exp.remove_docker_container()
            except Exception:
                pass
        return exp.api, exp.status

    def run_all(self):
        # Take a snapshot so we only run what existed at call time
        to_run = list(self.experiments)
        if self.num_parallel == 1:
            for exp in tqdm(to_run, desc="Running Experiments"):
                try:
                    self._run_experiment(exp)
                finally:
                    # Remove finished experiment to prevent re-execution
                    try:
                        self.experiments.remove(exp)
                    except ValueError:
                        pass
        else:
            with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
                with tqdm(total=len(to_run), desc="Running Experiments") as pbar:
                    future_to_exp = {executor.submit(self._run_experiment, exp): exp for exp in to_run}
                    try:
                        for future in as_completed(future_to_exp):
                            pbar.update(1)
                            exp = future_to_exp[future]
                            try:
                                future.result()
                            finally:
                                # Remove finished experiment to prevent re-execution
                                try:
                                    self.experiments.remove(exp)
                                except ValueError:
                                    pass
                    except KeyboardInterrupt:
                        print("\nInterrupted. Cancelling remaining experiments...")
                        for f in future_to_exp:
                            f.cancel()
                        raise

