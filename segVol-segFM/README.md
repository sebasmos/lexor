
✅ env

```
conda env create -f environment.yml
conda activate segvol
```

weights:
```
gdown 1dgM5slT5kDV3D_6k_vGpGqU5yB1nTwCL
```

⸻

✅ 2. predict.sh

Make sure this file is in your project folder:
```
#!/bin/bash
echo "Running Lexor-Coreset Inference"
python infer_case.py
```
Then make it executable:
```
chmod +x predict.sh
```

⸻

✅ 3. Edit infer_case_docker.py (ensure these lines are present)

Replace the input/output path lines like this:
```
npz_files = glob("/workspace/inputs/*.npz")
out_dir = "/workspace/outputs"
```
⸻

✅ 4. Build, Save, and Run the Container

🔧 Build the Docker image:

`docker build -t lexor_coreset:latest .`

📦 Save it as a `.tar.gz` archive:

`docker save lexor_coreset:latest | gzip > lexor_coreset.tar.gz`

🧪 Run the container using the command style in your screenshot:

`docker load -i lexor_coreset.tar.gz`

```
docker container run --gpus "device=0" -m 32G --name lexor_coreset --rm \
-v $PWD/imgs/:/workspace/inputs/ \
-v $PWD/lexor_outputs/:/workspace/outputs/ \
lexor_coreset:latest /bin/bash -c "sh predict.sh"
```


⸻
