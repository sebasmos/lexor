define output dir where data will be stored
define the input dir so just the anme fo the folder i wanna download, instead of the fixed one below from huggingface_hub import login

# Replace 'your-access-token' with your actual Hugging Face token
login(token="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")from huggingface_hub import hf_hub_download, list_repo_files
import os

repo_id = "junma/CVPR-BiomedSegFM"
subfolder = "3D_val_npz"

os.makedirs(subfolder, exist_ok=True)

files = list_repo_files(repo_id, repo_type="dataset")
val_npz_files = [f for f in files if f.startswith("3D_val_npz/") and f.endswith(".npz")]

for filename in val_npz_files:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=subfolder
    )
    print(f"Downloaded: {filename}")
 so the repo idiis the same but i wanna change subfolder = "3D_val_npz"
 by 3D_train_npz_random_10percent_16G
