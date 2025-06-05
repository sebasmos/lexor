# **Lexor**

---

## ✅ Environment Setup

```bash
conda env create -f environment.yml
conda activate segvol
```

Download pretrained weights:

```bash
gdown 1dgM5slT5kDV3D_6k_vGpGqU5yB1nTwCL
```

---

## ✅ Training

Download the SegVol1 checkpoint:  
[📦 SegVol1 Drive Link](https://drive.google.com/file/d/1FPj_tiITss5vJF91SrfPEURH6CUEmo4u/view)

Launch training with:
```bash
torchrun --nproc_per_node=2 train_fast.py
```

---

## ✅ Inference (Standalone)

1. Create `predict.sh`:
```bash
#!/bin/bash
echo "Running Lexor-Coreset Inference"
python infer_case.py
```

2. Make it executable:
```bash
chmod +x predict.sh
```

3. In `infer_case_docker.py`, modify the input/output paths:
```python
npz_files = glob("/workspace/inputs/*.npz")
out_dir = "/workspace/outputs"
```

---

## ✅ Docker Setup

### 🔧 Build Docker Image
```bash
docker build -t lexor_coreset:latest .
```

### 📆 Save Image
```bash
docker save lexor_coreset:latest | gzip > lexor_coreset.tar.gz
```

### 🧪 Run Inference Container
```bash
docker load -i lexor_coreset.tar.gz

docker container run --gpus "device=0" -m 32G --name lexor_coreset --rm \
  -v $PWD/imgs/:/workspace/inputs/ \
  -v $PWD/lexor_outputs/:/workspace/outputs/ \
  lexor_coreset:latest /bin/bash -c "sh predict.sh"
```

---

## 🧚 Alternate Predict Pipeline (manual test)

```bash
python predict.py
```

*Note: `predict_fast.py` was an experimental script and is not recommended.*

---

## 🐳 Advanced Docker Use (Custom Weights, Interactive Debugging)

### Build from `segVol-segFM` folder
```bash
docker build -t lexor:latest .
```

Update paths in the container as needed:

```bash
docker container run --gpus "device=0" -m 8G --name lexor --rm \
  -v $PWD/../../demo_cases:/workspace/inputs/ \
  -v $PWD/../outputs/:/workspace/outputs/ \
  -v $PWD/../weights/:/workspace/weights/ \
  lexor:latest
```

Or interactively:
```bash
docker container run --gpus "device=0" -m 8G --name lexor --rm \
  -v $PWD/../../demo_cases:/workspace/inputs/ \
  -v $PWD/../outputs/:/workspace/outputs/ \
  -it lexor:latest bash
```

---

## 🧚 Final Prediction and Evaluation

### Run Predictions
```bash
docker container run --gpus "device=0" -m 8G --name lexor --rm \
  -v $PWD/../mini_version/3D_val_npz:/workspace/inputs/ \
  -v $PWD/../outputs/:/workspace/outputs/ \
  lexor:latest /bin/bash predict.sh
```

### Evaluation
Navigate to the `evaluation/medseg` folder and run:
```bash
python CVPR25_iter_eval.py \
  -i ../../mini_version/3D_val_npz/ \
  -val_gts ../../mini_version/3D_val_gt/ \
  -o final_op \
  -d ../../docker_images/
```

---