# Lexor submission


`train_fast2.py`: Implementing fast encoders

`train_fast3.py`: commenting out validation phase

✅ env

```
conda env create -f environment.yml
conda activate segvol
```

weights:
```
gdown 1dgM5slT5kDV3D_6k_vGpGqU5yB1nTwCL
```

✅ Training

- Get SegVol1 checkpoint: [Drive](https://drive.google.com/file/d/1FPj_tiITss5vJF91SrfPEURH6CUEmo4u/view)

`torchrun  --nproc_per_node=2 train_fast.py`

- Fast encoders versiom

`torchrun --nproc_per_node=2   train_fast2.py     --fast_encoder_type mobilenet_2_5d     --batch_size 1     --save_dir "./ckpts_mobilenet_2_5d"     --num_epochs 3000     --initial_lr 1e-5     --train_root "/home/sebastian/codes/repo_clean/luxor-cvpr_own/segVol-segFM/3D_train_npz_random_10percent_16G"     --resume_ckpt "/home/sebastian/codes/repo_clean/luxor-cvpr_mygit/segVol-segFM/ckpts_mobilenet_2_5d/epoch_50_loss_0.6235_mobilenet_2_5d.pth"     --model_dir "./segvol"`

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
