import prefect
from prefect import flow, task
import subprocess
import os

@task
def run_training_script():
    """Runs the training script and captures its output."""
    print("Running model training script...")
    result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)
    print("Training script finished.")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    return result

@flow(name="Model Retraining Flow")
def retraining_flow():
    """Defines a simple Prefect flow for model retraining."""
    run_training_script()

if __name__ == "__main__":
    retraining_flow()