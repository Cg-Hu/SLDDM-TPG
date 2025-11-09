# **SLDDM-TPG: Semi-supervised Latent Disentangled Diffusion Model for Textile Pattern Generation**

## **Framework**
Our method targets generation of pattern images from clothing images (TPG) and consists of two stages: (1) a latent disentangled network (LDN); (2) a semi-supervised latent diffusion model (S-LDM).
<div align="center">
  <img src="./assets/framework.jpg" alt="framework" width="70%">
</div>

                         



## **Installation**

**Create conda environment**

```
conda env create -f environment.yaml
conda activate slddm
```

## **Usage**

### **Download Pretrained Models**

We use stable-diffusion-v1-5 as our backbone and you can download the public weights.

We use SimSiam (original model: resnet18, is public) as our LDN's SCM model, and the remaining modules are trained from weight initialization.

### **Download the Dataset**

Our CTP-HD dataset will be made public soon. Some cases are shown below.
<div align="center">
  <img src="assets/CTP-HD.jpg" alt="framework" width="50%">
</div>
VITON-HD data is publicly available and you can obtain and download it.

For the downloaded data, we recommend storing it in three hierarchical directories: *cloth, cloth_mask, and pattren*. cloth is the clothing image, cloth_mask is the mask image of the clothing image, and pattern is the textile pattern image corresponding to the clothing image. 

### **Training for Stage1: LDN**

You can train the LDN network by executing the following script. Here, `accumulate_steps` means accumulating 8 batches of gradients before back propagation to simulate the effect of large batch training.

```python
CUDA_VISIBLE_DEVICES=<set usable cuda index, such as 0,1> python main_sd.py \
  --dist-url 'tcp://localhost:12356' \
  --multiprocessing-distributed \
  --world-size 2 \
  --rank 0 \
  --resume <firstly train, please set to None>
  --scm-epochs 100 \
  --epochs 200 \
  --weight-decay 1e-3 \
  --learning-rate 1e-4 \
  --final-lr 1e-6 \
  --fix-pred-lr \
  --batch-size 32 \
  --accumulate_steps 8 \
  <add your dataroot_path, such as /home/name/ctp_hd_data>
```

### **Training for Stage2: S-LDM**

You can set trainer-related parameters in `SLDDM-TPG/configs/slddm512.yaml`, modify `gpus` to indicate the GPU number you use, and the code supports distributed training. 

```python
python -u main.py \
	--logdir logs/train_106 \
	--pretrained_model <set the downloaded stable-diffusion-v1-5 path> \
	--base config/slddm512.yaml \
	--scale_lr False
```

### **Run Inference**

Inference can adjust the sampling steps of DDIM. You can use the following script to perform inference:

```python
python test.py --gpu_id 0 \
--ddim_steps 100 \
--outdir <The directory where the generated results are stored> \
--config config/slddm512.yaml \
--dataroot <Directory of test clothing images and masks> \
--ckpt <Directory of ckpt obtained through training>
--n_samples 4 \
--seed 23 \
--scale 1 \
--H 512 \
--W 512 \
```