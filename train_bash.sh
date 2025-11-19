python -m torch.distributed.launch  --use_env train_video.py --ndevice 8 --output_len 16 
# python train_video.py --ndevice 1 --output_len 16 