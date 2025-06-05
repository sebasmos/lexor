from huggingface_hub import HfApi, login

# Log in to Hugging Face (optional if token is set in environment)
login(token="xxx")

# Initialize the Hugging Face API
api = HfApi()

# Upload the file to the dataset repository
repo_id = "sebasmos/16gCVPR"  # Your dataset repository
path_in_repo = "lexor_coreset.tar.gz"  # Destination path in the repo
local_file_path = "./lexor_coreset.tar.gz"  # Local path to your file

api.upload_file(
    path_or_fileobj=local_file_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
)

print(f"File {path_in_repo} uploaded to {repo_id}")
