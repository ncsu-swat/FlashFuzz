import enum
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import glob
import shutil
import time
from typing import Callable, Optional

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
    def __init__(self, dll: str, mode: str, ver: str, api: str, cpus: int = 2, mem: int = 16, check_valid: bool = False, time_budget: int = 180, itv: int = 60):
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
        self.container_name = f"{self.api}_container"
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
        print(f"Image {self.image_name} is available.")

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
        cmd = f'docker create --name {self.api}_container {self.image_name}'
        try:
            proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self.container_id = proc.stdout.strip()
            print(f"Created container {self.container_name} with ID: {self.container_id}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create container {self.container_name}.")
            print(f"Stderr: {e.stderr}")
            raise

    def start_docker_container(self):
        cmd = f"docker run -itd --cpus {self.cpus} -m {self.mem}g --name {self.container_name} {self.image_name}"
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

    def remove_docker_container(self):
        cmd = f"docker rm {self.container_name}"
        try:
            subprocess.run(cmd, shell=True, check=True)
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
            self.execute_command(f"cd /root/tensorflow/fuzz/ && python3 build_test_harness.py --dll {self.dll} --mode {self.mode} --ver {self.ver} --time_budget {self.time_budget}")
            self.execute_command(f"cd /root/tensorflow/fuzz/{self.api} && bash fuzz.sh > execution.log")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}/execution.log", self.result_dir)
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}/fuzz-0.log", self.result_dir)
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}/corpus", self.result_dir)
            self.status = Status.COMPLETED
        except Exception:
            self.status = Status.FAILED

    def tf_check_valid(self):
        self.status = Status.RUNNING
        try:
            self.check_image()
            self.start_docker_container()
            self.execute_command(f"cd /root/tensorflow/fuzz/ && python3 build_test_harness.py --dll {self.dll} --mode {self.mode} --check_valid ")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}", f"{self.result_dir}/{self.api}")
            os.makedirs(f"{self.result_dir}/build_status", exist_ok=True)
            self.copy_results_from_container(f"/root/tensorflow/fuzz/success_apis.txt", f"{self.result_dir}/build_status/")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/fail_apis.txt", f"{self.result_dir}/build_status/")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/build_summary.txt", f"{self.result_dir}/build_status/")
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
            self.copy_files_to_container(f"./_fuzz_result/{self.dll}{self.ver}-fuzz-{self.time_budget}s/{self.api}/corpus_itv_{self.itv}", f"/root/tensorflow/fuzz/{self.api}/corpus_itv_{self.itv}")
            self.execute_command(f"cd /root/tensorflow/fuzz/{self.api} && python3 generate_coverage_file.py --dll tf --itv {self.itv} --api {self.api}")
            self.copy_results_from_container(f"/root/tensorflow/fuzz/{self.api}/corpus_itv_{self.itv}", f"{self.result_dir}")
            loop_until_ctrl_c()
            self.status = Status.COMPLETED
        except Exception:
            self.status = Status.FAILED

    def run(self):
        if self.dll == "tf":
            if self.check_valid:
                self.tf_check_valid()
            elif self.mode == "fuzz":
                self.tf_fuzz()
            elif self.mode == "cov":
                if self.api != "all":
                    self.tf_cov_api()

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
        if self.num_parallel == 1:
            for exp in tqdm(self.experiments, desc="Running Experiments"):
                self._run_experiment(exp)
            return
        else:
            with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
                with tqdm(total=len(self.experiments), desc="Running Experiments") as pbar:
                    futures = [executor.submit(self._run_experiment, exp) for exp in self.experiments]
                    try:
                        for future in as_completed(futures):
                            pbar.update(1)
                            future.result()
                    except KeyboardInterrupt:
                        print("\nInterrupted. Cancelling remaining experiments...")
                        for f in futures:
                            f.cancel()
                        raise
