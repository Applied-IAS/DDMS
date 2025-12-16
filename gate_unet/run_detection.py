# -*- coding: utf-8 -*-
import os
import cv2
import time
import argparse
import numpy as np
from PIL import Image
from functools import cmp_to_key

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from gated_s_unet import unet_seg
from config_gated import *

from fusion import write_gifs

# ----------------------------
#  Argument Parser
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str,
                    default='/mnt/daikuai/nature_satellite_data_test/test_202204/',
                    help='input predicted sequence folder')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ----------------------------
#   Random Seed
# ----------------------------
torch.manual_seed(2211)
torch.cuda.manual_seed_all(2211)
np.random.seed(2211)
torch.backends.cudnn.deterministic = True

# ----------------------------
#   Filename Sorting
# ----------------------------
def model_name_num(s):
    return int(s.split('.')[0])

def cmp_name(x, y):
    return -1 if model_name_num(x) < model_name_num(y) else (1 if model_name_num(x) > model_name_num(y) else 0)

# ----------------------------
#   Load Frames from Folder
# ----------------------------
def make_dataset(dataset_dir):
    frame_path = []
    for folder in sorted(os.listdir(dataset_dir), key=cmp_to_key(cmp_name)):
        full = os.path.join(dataset_dir, folder)
        if not os.path.isdir(full):
            continue
        imgs = sorted(os.listdir(full), key=cmp_to_key(cmp_name))
        frame_path.append([os.path.join(full, x) for x in imgs])
    return frame_path

# ----------------------------
#   Satellite Dataset
# ----------------------------
class Satellite(Dataset):
    def __init__(self, dataset_dir, seq_len, train=True):
        self.frame_path = make_dataset(dataset_dir)
        self.seq_len = seq_len
        self.train = train
        self.clips = []

        for vid, frames in enumerate(self.frame_path):
            n = len(frames)
            if train:
                self.clips += [(vid, t) for t in range(n - seq_len + 1)]
            else:
                self.clips += [(vid, t * seq_len) for t in range(n // seq_len)]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vid, start = self.clips[idx]
        seq = []

        for t in range(start, start + self.seq_len):
            img = cv2.imread(self.frame_path[vid][t], cv2.IMREAD_GRAYSCALE)
            img = self.pad(img)
            img = img.astype(np.float32) / 255.0
            seq.append(np.expand_dims(img, 0))

        return torch.FloatTensor(seq)

    @staticmethod
    def pad(img):
        img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        pad = nn.ReflectionPad2d((0, 0, 19, 19))
        return pad(img).squeeze().numpy()

# ----------------------------
#   Load Model & Loss
# ----------------------------
model = unet_seg(in_ch=CHANNEL_IN, out_ch=CHANNEL_OUT)
model = nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_NORM)

alpha = torch.Tensor(LOSS_WEIGHT)

# ----------------------------
#   Safe Checkpoint Loading
# ----------------------------
def load_checkpoint(model, ckpt, optimizer):
    print(f'loading checkpoint: {ckpt}')
    ckpt_data = torch.load(ckpt, weights_only=False)
    model.load_state_dict(ckpt_data['state_dict'])
    optimizer.load_state_dict(ckpt_data['optimizer'])
    return model, optimizer

if checkpoint_file_pre is not None:
    model, optimizer = load_checkpoint(model, checkpoint_file_pre, optimizer)

model.eval()

# ----------------------------
#   Tools
# ----------------------------
def read_imgs(dir):
    imgs = []
    for i in range(16):
        img = cv2.imread(os.path.join(dir, f'{i}.png'), cv2.IMREAD_GRAYSCALE)
        imgs.append(img / 255.0)
    return imgs

# ----------------------------
#   Start Test
# ----------------------------
length = len(os.listdir(args.load_path))

colormap = [
    (130, 0, 102), (130, 0, 102), (130, 0, 102),
    (212, 0, 37), (255, 35, 2), (245, 175, 15),
    (255, 226, 3), (255, 226, 3), (255, 226, 3), (255, 226, 3)
]
valuemap = [190 + 5 * k for k in range(9)]

with torch.no_grad():
    for idx in range(length):
        print(f"Testing sequence {idx}/{length}")
        imgs = read_imgs(os.path.join(args.load_path, str(idx)) + '/')

        b_x = torch.FloatTensor(imgs).unsqueeze(0).unsqueeze(2).cuda()

        for frame_i in range(16):
            pred, edge = model(b_x[:, frame_i])

            pred = torch.argmax(pred.permute(0, 2, 3, 1), dim=3)
            pred = pred.cpu().numpy()
            pred = pred[:, 19:19+730, :]

            raw = b_x[:, frame_i].cpu().numpy()[0, 0, 19:19+730] * 255.0
            r, c = raw.shape

            save_dir = os.path.join(args.load_path, str(idx), 'seg_new_bar')
            os.makedirs(save_dir, exist_ok=True)

            if SAVE_IMAGE == 1:
                for b_i in range(len(pred)):
                    mask = pred[b_i] * 255
                    lw = 320 - (raw * (320 - 180) / 255.0)

                    out = np.zeros((r, c, 4), dtype=np.uint8)
                    out[..., :3] = np.stack([mask, mask, mask], axis=-1)
                    out[..., 3] = (mask > 0).astype(np.uint8) * 255

                    idx_mask = mask == 255
                    out[idx_mask] = (0, 0, 139, 255)

                    for k, th in enumerate(valuemap):
                        r0, g0, b0 = colormap[k]
                        idx2 = lw < th
                        out[idx2, :3] = (r0, g0, b0)
                        lw[idx2] = 9999  # mark visited

                    Image.fromarray(out, mode="RGBA").save(os.path.join(save_dir, f"{frame_i}.png"))

write_gifs(args.load_path)