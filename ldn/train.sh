CUDA_VISIBLE_DEVICES=0,1 python main_sd.py \
  --dist-url 'tcp://localhost:12355' \
  --multiprocessing-distributed \
  --world-size 2 \
  --rank 0 \
  --scm-epochs 100 \
  --epochs 200 \
  --weight-decay 1e-3 \
  --learning-rate 1e-4 \
  --final-lr 1e-6 \
  --fix-pred-lr \
  --batch-size 32 \
  --accumulate_steps 8 \
  /nfs5/hcg/datasets/VITON-HD-V2