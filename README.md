# Multi-Conditional Image Generation

## Setup

### 1. Create and Activate Conda Environment
```bash
conda env create -f environment.yaml
conda activate ldm
```

### 2. Download Pre-Trained Models
Download the models from the links provided below and place them in the appropriate directories.

### 3. Run the Model
```bash
python main.py
```

## Configuration
Modify the `config.py` file to change parameters and generate images based on desired conditions.

## Results
Generated images can be found in:
```bash
./exp/image_samples/multi_cond
```

## Models

### 1. Human Face Diffusion Model (SDEdit)
- Place the model in: `./exp/logs/celeba/celeba_hq.ckpt`
- [Download](https://huggingface.co/gwang-kim/DiffusionCLIP-CelebA_HQ/tree/main)

### 2. Unconditional Guided Diffusion Model
- Place the model in: `./exp/logs/imagenet/256x256_diffusion_uncond.pt`
- [Download](https://github.com/openai/guided-diffusion)

### 3. Face Parsing Model
- Place the model in: `./functions/face_parsing/79999_iter.pth`
- [Download](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) ([Repo](https://github.com/zllrunning/face-parsing.PyTorch))

### 4. Sketch Model
- Place the model in: `./functions/anime2sketch/netG.pth`
- [Download](https://drive.google.com/drive/folders/1Srf-WYUixK0wiUddc9y3pNKHHno5PN6R)

### 5. Landmark Model
- Place the models in:
  - `./functions/landmark/checkpoint/mobilefacenet_model_best.pth.tar`  
    [Download](https://github.com/cunjian/pytorch_face_landmark/blob/master/checkpoint/mobilefacenet_model_best.pth.tar)
  - `./functions/landmark/checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar`  
    [Download](https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view)
  - `./functions/landmark/Retinaface/weights/mobilenet0.25_Final.pth`  
    [Download](https://github.com/cunjian/pytorch_face_landmark/tree/master/Retinaface/weights)

### 6. ArcFace Model
- Place the model in: `./functions/arcface/model_ir_se50.pth`
- [Download](https://onedrive.live.com/?authkey=%21AOw5TZL8cWlj10I&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21835&parId=root&action=locate) ([Repo](https://github.com/paul-pias/Face-Recognition?tab=readme-ov-file))

---

## Troubleshooting

### Conda Activation Issue
If you encounter the following error repeatedly:
```bash
CondaError: Run 'conda init' before 'conda activate' for 'conda activate ldm'
```
Try the following steps:
1. Open PowerShell as Administrator.
2. Run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

   & C:/ProgramData/miniconda3/shell/condabin/conda-hook.ps1

   conda activate ldm
   ```