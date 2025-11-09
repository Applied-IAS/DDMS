# IMAGE_DIR = '../convection_data/bright_images/'
# IMAGE_DIR_REAL = '../convection_data/bright_images/'
# IMAGE_DIR = '/extend/shixc/data_1/bright_images/'  # 亮温图像路径
IMAGE_DIR = '/extend/shixc/bright_images/'  # 亮温图像路径
# LABEL_DIR = '../convection_data/labels/all/'  # 标注mask路径
LABEL_DIR = '/extend/shixc/labels_v2/'  # 标注mask路径
# LABEL_DIR = '/extend/shixc/data_1/labels_0825/' #output_0825
# LABEL_DIR = '/extend/shixc/data_1/labels_v3_1002/'
# LABEL_DIR = '/extend/shixc/data_2020/label_vis/'  # 可见光时段的标注mask路径
# IF2_DIR = '/extend/shixc/data_2020/IF2_new_img/' # 6.2通道数据
# VIS_DIR = '/extend/shixc/data_2020/vis_no_black/' # 可见光通道规定时间段数据

# 共用参数
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
TIME_START = '20200703_102846'
# LAST_EPOCH = 32 - 1
# checkpoint_file_pre = None
# TIME_START = None
# LAST_EPOCH = 0 - 1
SAVE_PERIOD = 10  # epoch % SAVE_PERIOD == 0时保存验证集识别结果
# LOSS_WEIGHT = [[0.25], [0.75]]  # 负样本和正样本loss的权重 [[0.25], [0.75]], [[0.2], [1]], None([[1], [1]])
# LOSS_WEIGHT = [[0.2], [1]]
# LOSS_WEIGHT = [[0.75], [1]]
# LOSS_WEIGHT = [[1], [1]]
# LOSS_WEIGHT = [[1.2], [0.8]]
# LOSS_WEIGHT = [[0.75], [1.25]]
# LOSS_WEIGHT = [[0.5], [1.5]]
# LOSS_WEIGHT = [[1], [1]]
LOSS_WEIGHT = [[0.25], [1]]
# LOSS_WEIGHT = [[0.5], [1]]

# 序列模型参数
# BATCH_SIZE = 2
# BATCH_SIZE_TEST = 1
# SEQ_LEN = 6
# PRED_LEN = 4
# LR = 0.001
# L2_NORM = 0.0

# 单帧模型参数
BATCH_SIZE = 16
BATCH_SIZE_TEST = 4
LR = 0.0001
L2_NORM = 0
BASIC_W = [2, 2, 1, 1, 1]

# 外推+分割模型
# BATCH_SIZE = 8
# BATCH_SIZE_TEST = BATCH_SIZE // 2
# IN_LEN = 4
# PRED_LEN = 4
# SEQ_LEN = 8
# LR = 0.0002
# L2_NORM = 0
# D_SIZE = 256

# 数据处理相关参数
LABEL_THRESHOLD = 50  # 整张图正样本像素数小于阈值的全置零
# CLIP_VAL = 40  # 随机平移亮温变小的范围：随机变小[0, CLIP_VAL),导致小于0的clip为0
# SLOPE_MIN = 0.5  # 255->128 == 180K->250K
# IMAGE_WIDTH = 1280
# IMAGE_HEIGHT = 800  # 730
# COL_BOUNDARY = 448  # 东部雷达外区域[0:val, :], 置零则不截取
# ROW_BOUNDARY = 16   # 上下35空白padding裁切大小
# # 中北部雷达外区域 [0:row, 0:col]，置零则不截取
# MID_NORTH_ROW = 192
# MID_NORTH_COL = 800
# CROP_SIZE = 768
