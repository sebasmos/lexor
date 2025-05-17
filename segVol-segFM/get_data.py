from huggingface_hub import login, hf_hub_download, list_repo_files
import os

# Authenticate with Hugging Face
login(token="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")

# Define repository and target subfolder
repo_id = "junma/CVPR-BiomedSegFM"
subfolder = "3D_train_npz_random_10percent_16G"  # CHANGE THIS to the folder you want

# Make local output directory
os.makedirs(subfolder, exist_ok=True)

# List and filter relevant .npz files from the specified subfolder
files = list_repo_files(repo_id, repo_type="dataset")
target_files = [f for f in files if f.startswith(f"{subfolder}/") and f.endswith(".npz")]

# Download each file
for filename in target_files:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=subfolder
    )
    print(f"Downloaded: {filename}")
