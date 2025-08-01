import enum
import subprocess

class Status(enum.Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Experiment():
    def __init__(self, dll: str, mode: str, ver: str, api: str, cpus: int = 2):
        self.dll = dll
        self.mode = mode
        self.ver = ver
        self.api = api
        self.cpus = cpus
        self.status = Status.NOT_STARTED
        self.image_name = f"ncsuswat/flashfuzz:{self.dll}{self.ver}-{self.mode}"
        self.container_name = f"{self.api}_container"
        self.container_id = None
        self.result_dir = f"./_fuzz_result/{self.dll}{self.ver}-{self.mode}/{self.api}_fuzz_output"
        subprocess.run(f"mkdir -p {self.result_dir}", shell=True, check=True)
        self.check_image()

    def check_image(self):

        cmd = f"docker image -q {self.image_name}"
        proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        out, _ = proc.communicate()
        image_available = out.decode().strip() != ""
        
        if not image_available:
            print(f"Image {self.image_name} not found. Please build it first.")
            return False
        print(f"Image {self.image_name} is available.")
        return True

    def create_docker_container(self):
        cmd = f'docker create --name {self.api}_container {self.image_name} --cpus="{self.cpus}"'
        try:
            proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self.container_id = proc.stdout.strip()
            print(f"Created container {self.container_name} with ID: {self.container_id}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create container {self.container_name}.")
            print(f"Stderr: {e.stderr}")
            raise
        
    def start_docker_container(self):
        cmd = f"docker start {self.container_name}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Started container {self.container_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start container {self.container_name}.")
            print(f"Stderr: {e.stderr}")
            raise

    def execute_command(self, command: str):
        cmd = f"docker exec {self.container_name} {command}"
        try:
            proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"Command executed successfully: {command}")
            return proc.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Command execution failed: {command}")
            print(f"Stderr: {e.stderr}")
            raise

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

    def copy_results(self, src: str, dest: str):
        cmd = f"docker cp {self.container_name}:{src} {dest}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Copied results from {src} to {dest}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to copy results from {src} to {dest}.")
            print(f"Stderr: {e.stderr}")
            raise

    def run(self):
        if self.dll == "tf" and self.mode == "fuzz":
            self.status = Status.RUNNING
            try:
                self.create_docker_container()
                self.start_docker_container()
                self.execute_command(f"cd /root/tensorflow/fuzz/{self.api} && bash fuzz.sh > execution.log")
                self.copy_results(f"/root/tensorflow/fuzz/{self.api}/execution.log", self.result_dir)
                self.copy_results(f"/root/tensorflow/fuzz/{self.api}/fuzz-0.log", self.result_dir)
                self.copy_results(f"/root/tensorflow/fuzz/{self.api}/corpus", self.result_dir)
                self.status = Status.COMPLETED
            except subprocess.CalledProcessError:
                self.status = Status.FAILED

class Scheduler():
    def __init__(self):
        self.experiments: list[Experiment] = []

    def add_experiment(self, experiment: Experiment):
        self.experiments.append(experiment)

    def run_all(self):
        for exp in self.experiments:
            print(f"Running experiment for {exp.api}...")
            exp.run()
            if exp.status == Status.COMPLETED:
                print(f"Experiment for {exp.api} completed successfully.")
            else:
                print(f"Experiment for {exp.api} failed.")
            exp.stop_docker_container()
            exp.remove_docker_container()
