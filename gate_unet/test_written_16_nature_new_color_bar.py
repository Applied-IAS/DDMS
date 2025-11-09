# -*- coding: utf-8 -*-
from  PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from tensorboard_logger import Logger
# from unet_model import UNet
from gated_s_unet import unet_seg, multi_loss_layer
import os
import cv2
import numpy as np
import time
from my_exception import *
from config_gated import *
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from functools import cmp_to_key

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default='/mnt/daikuai/nature_satellite_data_test/test_202204/',
                    help='training dataset (movingmnist or kth)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES']='2'

# from statistic.evaluate import calculate as cal_sig
# from statistic.eval import calculate as cal_seq
from dataset_gated_self import UnetDataset, DugsDataset
# from dataset_self_def import PredDataset as MyDataset
from losses import FocalLoss, JointEdgeSegLoss, SoftDiceLoss
from metrics import merge_img, iou_metric, pa_metric, pod_metric, far_metric, csi_metric

torch.manual_seed(2211)
torch.cuda.manual_seed_all(2211)
np.random.seed(2211)
torch.backends.cudnn.deterministic = True

if TIME_START is None:
    time_start = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
else:
    time_start = TIME_START

# tenb_logger = Logger(logdir="./tensorboard_logs", flush_secs=10)
checkpoint_path = './checkpoints/{}/'.format \
    (time_start)
# checkpoint_path = '/extend/RemoteSensingProject/ResultData/FY3B_VIRR_North_Africa_DL/result/mst_disturb_label_vggm_avg_pool_single_channel_t_20191031_184354_sz_512_lr_5e-05_bs_64_ch_2/'
logs_path = checkpoint_path + 'logs.txt'

# train_data = UnetDataset(label_dir=LABEL_DIR, image_dir=IMAGE_DIR, mode='train')
# train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
#
# val_data = UnetDataset(label_dir=LABEL_DIR, image_dir=IMAGE_DIR, mode='val')
# val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=4)
def model_name_compare(x, y):
    x = deal_name(x)
    y = deal_name(y)
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0

def deal_name(s):
    s = s.split('.')[0]
    # names = s.split('_')
    # print(names)
    # return int(names[-2])*100 + int(names[-1])
    return int(s)
        

def make_dataset(dataset_dir):
    frame_path = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(sorted(os.listdir(dataset_dir), key = cmp_to_key(model_name_compare))):
        # clipsFolderPath = os.path.join(dataset_dir, folder, 'ground_truth')
        clipsFolderPath = os.path.join(dataset_dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        frame_path.append([])
        # Find and loop over all the frames inside the clip.
        
        for image in sorted(os.listdir(clipsFolderPath), key = cmp_to_key(model_name_compare)): #这里不排序的话，序列就被打断了
        # for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            
            # if int(image.split('.')[0]) <41:
            #     print(image)
            frame_path[index].append(os.path.join(clipsFolderPath, image))
            
    return frame_path

class Satellite(Dataset):
    def __init__(self, dataset_dir, seq_len, train=True):
        self.frame_path = make_dataset(dataset_dir)
        self.seq_len = seq_len
        self.train = train
        # print(len(self.frame_path))
        self.clips = []
        for video_i in range(len(self.frame_path)):
            video_frame_num = len(self.frame_path[video_i])
            self.clips += [(video_i, t) for t in range(video_frame_num - seq_len + 1)] if train \
                else [(video_i, t * seq_len) for t in range(video_frame_num // seq_len)]

    def __getitem__(self, idx):
        (video_idx, data_start) = self.clips[idx]
        sample = []
        for frame_range_i in range(data_start, data_start+self.seq_len):
            frame = cv2.imread(self.frame_path[video_idx][frame_range_i], cv2.IMREAD_GRAYSCALE)
            frame = self.constant_padding_image(frame)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype(np.float32)
            frame = frame/255.0
            sample.append(frame)
        return torch.FloatTensor(sample)

    def constant_padding_image(self, image):
        # b, t, c, h, w = image.shape
        image = torch.FloatTensor(image)
        temp = image.unsqueeze(0).unsqueeze(0)
        # print(temp.shape)
        # print(temp.shape)
        m = torch.nn.ReflectionPad2d((0, 0, 19, 19))
        res = m(temp)
        # print(res.shape)
        # exit()
        return res.squeeze(0).squeeze(0).numpy()

    def __len__(self):
        return len(self.clips)

# test_dataset = Satellite('/mnt/A/daikuai/daikuai_new/10.248.19.24-files-backup/daikuai/LMC_satellite_img/china_satellite_images_202006_24', seq_len=24, train=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
# print(len(test_dataset))
model = unet_seg(in_ch=CHANNEL_IN, out_ch=CHANNEL_OUT)
# 并行
model = torch.nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_NORM)
# params_w = list(model.module.decoder_seg.dusample.conv_w.parameters())
# params_p = list(model.module.decoder_seg.dusample.conv_p.parameters())
# optimizer_w = torch.optim.SGD(params_w+params_p, lr=0.00008, momentum=0.9)
# optimizer_w = torch.optim.Adam(params_w+params_p, lr=0.0002, weight_decay=L2_NORM)

alpha = torch.Tensor(LOSS_WEIGHT)
print("alpha is {}".format(LOSS_WEIGHT))
seg_loss_func = FocalLoss(class_num=CHANNEL_OUT, alpha=alpha)
dice_loss = SoftDiceLoss()
joint_loss_train = JointEdgeSegLoss(classes=CHANNEL_OUT)
joint_loss_val = JointEdgeSegLoss(classes=CHANNEL_OUT, mode='val')
# 考虑建立个权重函数？越靠近冬天，误判为对流的损失越大？越靠近夏天，误判为非对流的损失越大？
# 函数的建立 通过累计每天 得到的标注的对流面积曲线拟合得到？
# 评价函数用每个连通区域是否命中来计算准确率？ 命中面 积多少不好整

miou_max = 0.6
loss_val_min = 0.1
best_snapshot = os.path.join(checkpoint_path, 'best_snapshot.pth.tar')
last_snapshot = os.path.join(checkpoint_path, 'last_snapshot.path.tar')


# device = torch.device("cpu")

def load_checkpoint(model, checkpoint_PATH, optimizer):
    print('loading checkpoint!')
    # model_CKPT = torch.load(checkpoint_PATH, map_location='cpu')
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    # loss_layer.load_state_dict(model_CKPT['loss_layer_dict'])
    optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer


if checkpoint_file_pre is not None:
    print("model is not null, model is loading from {}".format(checkpoint_file_pre))
    model, optimizer = load_checkpoint(model, checkpoint_file_pre, optimizer)
print("the mode of the learning is {}".format(MODE))


model.eval()
losses_val = []
miou = []
metric_dict = {'pa': pa_metric, 'pod': pod_metric, 'far': far_metric, 'csi': csi_metric}
result_dict = {'pa': [], 'pod': [], 'far': [], 'csi': []}
save_path = './202006_24_gt'
# save_path = '/home/lokfar/longys/web-data/mask/china'
#save_path = '/home/lokfar/longys/web-data/mask/south_east'
# save_path = '/extend/shixc/data_1/unet_0716/tensorboard_logs/new_test/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

def constant_padding_image(image):
    # b, t, c, h, w = image.shape
    image = torch.FloatTensor(image)
    temp = image.unsqueeze(0).unsqueeze(0)
    # print(temp.shape)
    # print(temp.shape)
    m = torch.nn.ReflectionPad2d((0, 0, 19, 19))
    res = m(temp)
    # print(res.shape)
    # exit()
    return res.squeeze(0).squeeze(0).numpy()

def read_imgs(dir):
    # path = dir + '/pred/'
    path = dir
    imgs = []
    for i in range(16):
        # print(path+str(i)+'.png')
        image =cv2.imread(path+str(i)+'.png', cv2.IMREAD_GRAYSCALE) / 255.0
        # image = constant_padding_image(image)
        imgs.append(image)
    return imgs
# length = 1374
# length = 58
# length = 58
length = len(os.listdir(args.load_path))
# load_path = '/home/ices/daikuai/RVD-satellite-nowcasting/evaluate/generated/resnet-adam-noise-l1-satellite-d64-t1000-residual-alFalse/pred/'
# load_path = '/mnt/daikuai/Satellite-Image-Sequence-Prediction-RVD-refine-local-motion-patterns./evaluate/generated/resnet-adam-noise-l1-satellite-d64-t1000-residual-alFalse/pred/'
# load_path = '/mnt/daikuai/Satellite-Image-Sequence-Prediction-RVD-refine-local-motion-patterns./evaluate/generated/resnet-adam-noise-l1-satellite-d64-t1000-residual-alFalse/pred/'

with torch.no_grad():
    for index_seq in range(length):
        # print('idx idx', idx)
        print("正在测试")
        imgs = read_imgs(args.load_path + str(index_seq) + '/')
        
        b_x = torch.FloatTensor(imgs).unsqueeze(0).unsqueeze(2).cuda()  # batch x, shape (batch, 28*28)
        # b_y = batch_x[1].cuda()  # batch y, shape (batch, 28*28)
        print(b_x.shape)
        for frame_i in range(16):
            print('this is the i-th frame', frame_i)
            pred, edge = model(b_x[:,frame_i,:,:,:])
            print(pred.shape)
            pred = pred.permute(0, 2, 3, 1)
            pred = torch.argmax(pred, dim=3)
            print(pred.shape)
            # 保存分割结果，真实图像和label
            pred = pred.cpu().numpy()
            print("Step: {}".format(index_seq))
            #pred = pred[:, 19:749, :]  #中国区参数
            # print("pred shape",pred.shape)
            #pred = pred[:,2:-2, 5:-5]  #东南亚参数
            # pred = pred[:, 14:1074, :]  # 中国区扩大后参数
            pred = pred[:, 19:19+730, :]
            
            imgTested = b_x[:,frame_i,:,:,:].cpu().numpy()
            # print('imgTested', imgTested.shape)
            # colormap = [(176, 23, 31), (227, 23, 13), (255, 0, 0), (255, 153, 18),
            # (227, 207, 87), (255, 255, 0), (56, 94, 15), (0, 255, 0),
            # (30, 144, 255), (0, 0, 139)]
            # 新的染色方案3 
            # colormap = [(255,226,3), (255,226,3), (255,226,3), (255,226,3), (245,175,15), (255,35,2), (212,0,37), (130,0,102),
            # (130,0,102), (130,0,102)]
            colormap = [(130,0,102), (130,0,102), (130,0,102), (212,0,37),  (255,35,2), (245,175,15), (255,226,3),
             (255,226,3), (255,226,3), (255,226,3)]
            valuemap = [190 + 5 * k for k in range(9)]
            # 亮温阈值图片生成
            for b_i in range(len(pred)):
                print('this is a test', len(pred))
                # spilts = name[b_i].split('/')
                # mark_img_name = 'mark_' + spilts[-1]
                # img_save_path = name[b_i].replace(spilts[-1], mark_img_name)#/home/lokfar/lushenyuan/pythonProject/CNpred/20200601014500/pred/mark_1.png
                img_save_path = os.path.join(args.load_path + str(index_seq), 'seg_new_bar')
                if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path)

                if SAVE_IMAGE == 1:
                    black_img = pred[b_i] * 255
                    ori_img = imgTested[b_i][0,19:19+730] * 255.0

                    r, c = black_img.shape

                    new_img = np.zeros((r, c, 4), dtype=np.int8)

                    liangwen_img = np.zeros((r, c ,1),dtype=np.float32)
                    liangwen_img = 320.0 - (ori_img * (320.0-180.0)/255.0)

                    index = black_img == 255

                    new_img[:, :, 0] = black_img
                    new_img[:, :, 1] = black_img
                    new_img[:, :, 2] = black_img
                    new_img[:, :, -1] = np.where(black_img > 0, 255, 0)

                    new_img[index, 0] = 0
                    new_img[index, 1] = 0
                    new_img[index, 2] = 139


                    for i, thres in enumerate(valuemap):
                        r, g, b = colormap[i]
                        idx = liangwen_img < thres
                        new_img[idx, 0] = r
                        new_img[idx, 1] = g
                        new_img[idx, 2] = b
                        liangwen_img[idx] = 2211

                    # print(new_img.shape)
                    img = Image.fromarray(new_img, mode="RGBA")

                    img.save(os.path.join(img_save_path, str(frame_i)+'.png'))
                    # log.logger.info('生成mark:' + mark_img_name)


