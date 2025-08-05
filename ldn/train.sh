CUDA_VISIBLE_DEVICES=0,1 python main_sd.py \
  --dist-url 'tcp://localhost:12356' \
  --multiprocessing-distributed \
  --world-size 2 \
  --rank 0 \
  --resume /home/hcg/cloth_pattern/Representation/cpsd/logs_sd/106/2025-01-14T16:01:50/best/ckpt_epoch87_-0.21844900978936088.pth.tar \
  --scm-epochs 100 \
  --epochs 200 \
  --weight-decay 1e-3 \
  --learning-rate 1e-4 \
  --final-lr 1e-6 \
  --fix-pred-lr \
  --batch-size 16 \
  --accumulate_steps 8 \
  /nfs5/hcg/datasets/CTP-HD