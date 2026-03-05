
import os
from huggingface_hub import login
from datasets import load_dataset
import shutil

# Token provided by user
HF_TOKEN = os.getenv("HF_TOKEN") # "your_token_here"
DATASET_NAME = "AISE-TUDelft/Poisoned-Chalice"
OUTPUT_DIR = "poisoned_chalice_dataset"

# Set cache directory to D: drive to avoid space issues on C:
CACHE_DIR = "d:/Focus/Poisoned Chalice Competition 2026/.hf_cache"
TEMP_DIR = "d:/Focus/Poisoned Chalice Competition 2026/.tmp"

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR

def main():
    print(f"Logging in to Hugging Face...")
    login(token=HF_TOKEN)

    print(f"Downloading dataset {DATASET_NAME}...")
    # Load all 5 languages
    languages = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
    
    # Ensure output directory exists and is empty
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for lang in languages:
        print(f"Processing language: {lang}")
        # Download train and test splits
        ds = load_dataset(DATASET_NAME, lang)
        
        # Save to disk in a structured way that datasets can reload or just as parquet/json
        # Saving as HF dataset format (arrow) is best for reloading with load_from_disk
        save_path = os.path.join(OUTPUT_DIR, lang)
        ds.save_to_disk(save_path)
        print(f"Saved {lang} to {save_path}")

    print("Download complete. Zipping dataset...")
    # Create zip file
    shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
    
    print(f"Successfully created {OUTPUT_DIR}.zip")

if __name__ == "__main__":
    main()
