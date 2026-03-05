
import os
import subprocess
import sys
import glob

def run_command(command, check=True):
    """Run a shell command."""
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=check)

def main():
    # Configuration
    REPO_URL = "https://github.com/AISE-TUDelft/PoisonedChalice.git"
    REPO_DIR = "PoisonedChalice"
    MODEL = "bigcode/starcoder2-3b" # Or use other supported models
    SAMPLE_FRACTION = 0.01  # Use 1.0 for full run, smaller for testing
    OUTPUT_DIR = "results" # Relative to REPO_DIR
    PLOTS_DIR = "plots"   # Relative to REPO_DIR
    
    # HF Token Handling
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    print("=== Starting Poisoned Chalice Experiment Setup ===")

    # 1. Clone Repository (if not already exists)
    if not os.path.exists(REPO_DIR):
        print(f"Cloning {REPO_URL}...")
        run_command(f"git clone {REPO_URL}")
    else:
        print(f"Repository {REPO_DIR} already exists. Pulling latest changes...")
        # Use subprocess to change dir for git command to avoid global state issues
        subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

    # 2. Install Dependencies
    print("Installing dependencies...")
    # Install the package in editable mode
    run_command(f"pip install -e {REPO_DIR}")
    run_command("pip install accelerate>=1.12.0")

    # 3. Hugging Face Login
    if HF_TOKEN:
        print("HF_TOKEN found. Logging in...")
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN)
        except ImportError:
            run_command("pip install huggingface_hub")
            from huggingface_hub import login
            login(token=HF_TOKEN)
    else:
        print("WARNING: HF_TOKEN environment variable not found.")
        print("You may need to login manually or ensure your token is set for gated models.")

    # 4. Run Experiment
    print(f"Running experiment with model: {MODEL}")
    
    # We run run.py from within the REPO_DIR to ensure it finds local modules (MIAttack, etc.)
    # and writes results relative to itself.
    
    current_dir = os.getcwd()
    abs_repo_dir = os.path.abspath(REPO_DIR)
    
    try:
        os.chdir(abs_repo_dir)
        
        # Ensure output directory exists (run.py does this, but good to be safe)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        cmd = [
            sys.executable, "run.py",
            "--model_name", MODEL,
            "--attacks", "loss", "mkp", # Add "pac" if needed
            "--sample_fraction", str(SAMPLE_FRACTION),
            "--output_dir", OUTPUT_DIR,
            "--use_fp16"
        ]
        
        print(f"Executing in {abs_repo_dir}: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # 5. Process Results
        print("Processing results...")
        
        # Helper to find latest metadata
        search_path = os.path.join(OUTPUT_DIR, "metadata_*.json")
        metadata_files = glob.glob(search_path)
        
        if metadata_files:
            latest_config = max(metadata_files, key=os.path.getctime)
            print(f"Found latest config: {latest_config}")
            
            # Ensure plots directory exists
            if not os.path.exists(PLOTS_DIR):
                os.makedirs(PLOTS_DIR)
            
            # process.py arguments:
            # --config_path: path to metadata file
            # --results_folder: path to folder containing parquet files (needs trailing slash potentially)
            # --output_path: path to save plots (needs trailing slash potentially)
            
            results_folder_arg = os.path.join(OUTPUT_DIR, "") # Ensure trailing slash if needed by join
            plots_path_arg = os.path.join(PLOTS_DIR, "")
            
            process_cmd = [
                sys.executable, "process.py",
                "--config_path", latest_config,
                "--results_folder", results_folder_arg,
                "--output_path", plots_path_arg
            ]
            
            print(f"Executing: {' '.join(process_cmd)}")
            subprocess.check_call(process_cmd)
            
            print(f"Plots saved to {os.path.join(abs_repo_dir, PLOTS_DIR)}")
        else:
            print("No metadata file found. Experiment might have failed or produced no output.")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred during execution: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Always return to original directory
        os.chdir(current_dir)

    print("=== Experiment Finished ===")

if __name__ == "__main__":
    main()
