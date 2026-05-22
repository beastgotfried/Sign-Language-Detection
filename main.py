# main.py
import os
import subprocess
import sys

# --- Configuration ---
# Path to the python executable in the virtual environment
VENV_PYTHON = os.path.join("signlanguage", "Scripts", "python.exe")
# Path to the source directory
SRC_DIR = "src"

# --- Helper Functions ---
def get_python_executable():
    """
    Determines the correct Python executable to use.
    If a virtual environment is found and activated, it returns the path to the venv's Python.
    Otherwise, it returns the system's default Python executable.
    """
    if os.path.exists(VENV_PYTHON):
        print(f"Using Python from virtual environment: {VENV_PYTHON}")
        return VENV_PYTHON
    print(f"Virtual environment not found at {VENV_PYTHON}, using system Python: {sys.executable}")
    return sys.executable

def run_script(python_executable, script_name, script_args=None):
    """
    Runs a Python script using the specified interpreter and arguments.
    
    Args:
        python_executable (str): The path to the Python interpreter.
        script_name (str): The name of the script to run.
        script_args (list, optional): A list of command-line arguments for the script.
    """
    script_path = os.path.join(SRC_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    command = [python_executable, script_path]
    if script_args:
        command.extend(script_args)

    print(f"--- Running: {' '.join(command)} ---")
    try:
        process = subprocess.run(command, check=True, text=True, capture_output=True)
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print("Stderr:", process.stderr)
        print(f"--- Finished: {script_name} ---")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}.")
        print(f"Return code: {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        sys.exit(1)

def main():
    """
    Main function to run the sign language detection pipeline.
    """
    python_exec = get_python_executable()

    print("Starting the full pipeline...")

    # Step 1: Process images and create the feature dataset.
    print("\n[PHASE 1] Processing images to extract hand landmark features...")
    run_script(python_exec, "phase1_database.py")

    # Step 2: Train the model on the extracted features.
    print("\n[PHASE 2] Training the model...")
    run_script(python_exec, "phase3_train.py")

    # Step 3: Run inference using the trained model.
    print("\n[PHASE 3] Running inference...")
    # The inference script can be run in 'camera' or 'dataset' mode.
    # Defaulting to 'camera' mode here.
    run_script(python_exec, "phase4_inference.py", ["--mode", "camera"])

    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
