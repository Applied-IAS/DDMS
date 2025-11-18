# satellite nowcasting
python test_video.py --device 0

# convection detection
python ./gate_unet/run_detection.py --load_path '../results/evaluate/generated/resnet-adam-noise-l1-satellite-d64-t1000-residual-alFalse/pred/'