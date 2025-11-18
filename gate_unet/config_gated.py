# shared parameters
EPOCH = 20000
CHANNEL_IN = 1
CHANNEL_OUT = 2
MODE = 'test'  # train valid test
AREA = 'all' # north south all
SEQ = False # True False
if AREA == 'north' and SEQ is False:
    checkpoint_file_pre = './model_parameters/best-m-18-0.0002-0.0012-0.9211_north.pth.tar'
if AREA == 'south' and SEQ is False:
    checkpoint_file_pre = './model_parameters/best-m-51-0.0008-0.0056-0.9351_south.pth.tar'
if AREA == 'all' and SEQ is False:
    checkpoint_file_pre = './gate_unet/model_parameters/best-m-28-0.0006-0.0032-0.9255_all_area.pth.tar'
if AREA == 'all' and SEQ:
    checkpoint_file_pre = './dugs-unet-compare-with-rvos-params/best-m-20-0.0008-0.0029-0.9245-all-area.pth.tar'
SAVE_IMAGE = 1

SAVE_PERIOD = 10  
LOSS_WEIGHT = [[0.25], [1]]

BATCH_SIZE = 16
BATCH_SIZE_TEST = 4
LR = 0.0001
L2_NORM = 0
BASIC_W = [2, 2, 1, 1, 1]
LABEL_THRESHOLD = 50  

