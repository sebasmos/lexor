import os
import json

# Ppath to the folder containing your .npz validation files
npz_folder = "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/merged"

# Path where val_samples.json will be saved
output_json_path = "./val_samples.json"

# Collect all .npz files in the folder
val_samples = [
    os.path.join(npz_folder, fname)
    for fname in sorted(os.listdir(npz_folder))
    if fname.endswith(".npz")
]

# Save the list to a JSON file
with open(output_json_path, "w") as f:
    json.dump(val_samples, f, indent=2)

# Show basic info
print(f"val_samples.json has been saved at:\n{output_json_path}")
print(f"It contains {len(val_samples)} .npz files")

# Preview the first few entries
print("\n Preview:")
for entry in val_samples[:3]:
    print(f" - {entry}")
