from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
from torchvision import transforms
import edge_utils
import numpy as np
import time
import random
import random
import json
from config_gated import *
random.seed(2211)
np.random.seed(2211)

class UnetDataset(Dataset):

    def __init__(self, label_dir, image_dir, mode, seed=2019):

        '''
        # image目录结构： image_dir/年月/时间.png
        # label目录结构： label_dir/self.mode/年月/时间.png
        '''

        self.seed = seed
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        self.label_list = self.read_label_list(label_dir, image_dir, mode, seed)
        # if mode == 'train':
        #     self.label_list = self.label_list[:-1]
        self.len = len(self.label_list)
        print(self.label_list[-1], self.len)

    def __getitem__(self, i):
        index = i % self.len

        img, label = self.load_data(self.image_dir, self.label_dir, self.label_list[index], self.mode)
        img = self.data_preproccess(img)
        label = torch.LongTensor(label)

        if self.mode != 'train':
            return img, label, self.label_list[index]

        return img, label, self.label_list[index]
        # return img.cuda(), label.cuda()

    def __len__(self):
        return self.len


    # 未划分数据集
    def read_label_list(self, label_dir, image_dir, mode, seed):
        # 单帧模型训练数据集划分，小论文部分
        labels = []
        # for y_m in sorted(os.listdir(os.path.join(label_dir, 'all'))):
        for y_m in range(201801, 201813):
            labels_mon = sorted(os.listdir(os.path.join(label_dir, 'all', str(y_m))))
            # labels.extend(os.listdir(os.path.join(label_dir, 'all', y_m)))
            labels.append(labels_mon)
        # print(labels[0])

        # 判断对应的图像是否存在，不存在的移除
        labels_exist = []
        for month in range(len(labels)):
            labels_exist_mon = []
            for name in labels[month]:
                img_path = os.path.join(image_dir, name[:6], name[:-4] + '_bright_img.png')
                if os.path.exists(img_path):
                    labels_exist_mon.append(name)
            labels_exist.append(labels_exist_mon)
        if SEQ is False:
            labels_seperated = []
            for month in range(len(labels_exist)):
                len_seq_val = int(0.3 * len(labels_exist[month]))
                # if self.mode == 'train':
                #     # labels_seperated.extend(labels_exist[num_i*len_seq:(num_i+1)*len_seq-len_seq_val])
                #     # labels_seperated.extend(labels_exist[month][:-len_seq_val])
                #     labels_seperated.extend(labels_exist[month][:450])
                #     labels_seperated.extend(labels_exist[month][450+len_seq_val:])
                # elif self.mode == 'val':
                #     # labels_seperated.extend(labels_exist[month][-len_seq_val:-len_seq_val // 2])
                #     labels_seperated.extend(labels_exist[month][450: 450+(len_seq_val // 2)])
                # else:
                #     # labels_seperated.extend(labels_exist[(num_i+1)*len_seq-len_seq_val:(num_i+1)*len_seq])
                #     # labels_seperated.extend(labels_exist[month][-len_seq_val // 2:])
                #     labels_seperated.extend(labels_exist[month][450+(len_seq_val // 2): 450+len_seq_val])
                if self.mode == 'train':
                    # labels_seperated.extend(labels_exist[num_i*len_seq:(num_i+1)*len_seq-len_seq_val])
                    # labels_seperated.extend(labels_exist[month][:-len_seq_val])
                    # labels_seperated.extend(labels_exist[month][:200])
                    # labels_seperated.extend(labels_exist[month][200 + len_seq_val:])
                    if month == 3:
                        labels_seperated.extend(labels_exist[month][:-len_seq_val])
                    elif month == 5 or month == 7:
                        labels_seperated.extend(labels_exist[month][:0+(len_seq_val // 2)])
                        labels_seperated.extend(labels_exist[month][0+len_seq_val: -(len_seq_val // 2)])
                    else:
                        labels_seperated.extend(labels_exist[month][:450])
                        labels_seperated.extend(labels_exist[month][450 + len_seq_val:])
                elif self.mode == 'val':
                    # labels_seperated.extend(labels_exist[month][-len_seq_val:-len_seq_val // 2])\
                    # labels_seperated.extend(labels_exist[month][200: 200 + (len_seq_val // 2)])
                    if month == 3:
                        labels_seperated.extend(labels_exist[month][-len_seq_val:-len_seq_val // 2])
                    elif month == 5 or month == 7:
                        labels_seperated.extend(labels_exist[month][-(len_seq_val // 2):])
                    else:
                        labels_seperated.extend(labels_exist[month][450: 450+(len_seq_val // 2)])
                else:
                    # labels_seperated.extend(labels_exist[(num_i+1)*len_seq-len_seq_val:(num_i+1)*len_seq])
                    # labels_seperated.extend(labels_exist[month][-len_seq_val // 2:])
                    # labels_seperated.extend(labels_exist[month][200 + (len_seq_val // 2): 200 + len_seq_val])
                    if month == 3:
                        labels_seperated.extend(labels_exist[month][-(len_seq_val // 2):])
                    elif month == 5 or month == 7:
                        labels_seperated.extend(labels_exist[month][0+(len_seq_val // 2): 0+len_seq_val])
                    else:
                        labels_seperated.extend(labels_exist[month][450+(len_seq_val // 2): 450+len_seq_val])
        else:
            # 序列模型对比数据集划分，与rvos数据集划分相对应
            labels_seperated = []
            if self.mode == 'train':
                labels_seperated = json.load(open(os.path.join(label_dir, 'all', 'sequences_4_single_new_train.json'), 'r'))
            elif self.mode == 'val':
                labels_seperated = json.load(open(os.path.join(label_dir, 'all', 'sequences_4_single_new_val.json'), 'r'))
            else:
                labels_seperated = json.load(open(os.path.join(label_dir, 'all', 'sequences_4_single_new_test.json'), 'r'))

        if self.mode != 'test':
            np.random.seed(seed)
            np.random.shuffle(labels_seperated)

        return labels_seperated

    def load_data(self, image_dir, label_dir, name, mode):
        '''
        加载数据
        :return:
        '''
        # print(os.path.join(image_dir, name[:6], name))
        # print(os.path.join(label_dir, mode, name[:6], name))
        image = cv2.imread(os.path.join(image_dir, name[:6], name[:-4] + '_bright_img.png'), cv2.IMREAD_GRAYSCALE)
        # label = cv2.imread(os.path.join(label_dir, mode, name[:6], name), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(label_dir, 'all', name[:6], name), cv2.IMREAD_GRAYSCALE)
        if (label is None) | (image is None):
            print(name)
        # print(image.shape, label.shape)
        # label归一化
        # print(np.max(label))
        label = 1 * (label >= 200)

        # 去掉边界
        r, c = image.shape  # 800 * 1280
        r_b = 40
        c_b = 20
        if AREA == 'north':
            image = image[:400, 440:1240]
            label = label[:400, 440:1240]
            label[0:192, 0:355] = 0
            image[0:192, 0:355] = 0
        elif AREA == 'south':
            image = image[400:, 440:1240]
            label = label[400:, 440:1240]
        else:
            image = image[:, 440:1240]
            label = label[:, 440:1240]
            if SEQ is False:
                label[0:192, 0:355] = 0
                image[0:192, 0:355] = 0
        # 生成分割标签和边界标签
        labels = []
        _edgemap = edge_utils.mask_to_onehot(label + 1, 2)
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
        labels.append(label)
        labels.append(_edgemap[0])
        labels = np.array(labels)
        # 提取图像梯度，与原始图像进行stack
        # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        # gm = cv2.sqrt(sobel_x ** 2 + sobel_y ** 2).astype(np.uint8)
        # image = np.stack([image, gm], axis=0)



        # label[0:192, 0:355] = 0
        # image = cv2.resize(image, (730,730))
        # image = np.array(image, dtype=np.int)
        # np.random.seed(int(time.time() * 1000000) % 2 ** 32)
        # if mode == 'train':
        #     if int(np.random.randint(0, 2, 1)) == 1:
        #         image = np.flip(image, 0)
        #     if int(np.random.randint(0, 2, 1)) == 1:
        #         image = np.flip(image, 1)

        # 改变数据分布，减少对流目标区域对于亮温值得过度依赖
        # np.random.seed(int(time.time() * 1000000) % 2 ** 32)
        # type = np.random.randint(0, 2)
        # if type:
        #     # print(np.random.randint(30, 61))
        #     # print("Before: the mini pixel is {}, the maxi is{}".format(np.min(image), np.max(image)))
        #     np.random.seed(int(time.time() * 1000000) % 2 ** 32)
        #     image = image - np.random.randint(0, 31)
        #     image = np.clip(image, 0, 255)
        #     # print("After: the mini pixel is {}, the maxi is{}".format(np.min(image), np.max(image)))
        #
        # else:
        #     gray = 255 - np.max(image)
        #     np.random.seed(int(time.time() * 1000000) % 2 ** 32)
        #     image = image + np.random.randint(0, gray + 1)
        #     # print("the maxi pixel is {}".format(np.max(image)))

        r, c = image.shape
        image = np.reshape(image, (1, r, c)) / 255.0
        labels = np.reshape(labels, (2, r, c))
        labels[0] = labels[0] + 1

        return image.copy(), np.array(labels, dtype=int)

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        # data = self.toTensor(data)
        data = torch.Tensor(data)
        return data

class DugsDataset(Dataset):

    def __init__(self, label_dir, image_dir, mode, seed=2019):

        '''
        # image目录结构： image_dir/年月/时间.png
        # label目录结构： label_dir/self.mode/年月/时间.png
        '''

        self.seed = seed
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        self.label_list = self.read_label_list(label_dir, image_dir, mode, seed)
        # if mode == 'train':
        #     self.label_list = self.label_list[:-1]
        self.len = len(self.label_list)
        print("lennn  ",self.label_list[-1], self.len)
        # print("dsad  ",self.label_list[1], self.len)
        # print("lgfdgfd  ",self.label_list[1:10], self.len)

    def __getitem__(self, i):
        index = i % self.len

        img, label = self.load_data(self.image_dir, self.label_dir, self.label_list[index], self.mode)
        img = self.data_preproccess(img)
        label = torch.LongTensor(label)

        if self.mode != 'train':
            return img, label, self.label_list[index]

        return img, label, self.label_list[index]
        # return img.cuda(), label.cuda()

    def __len__(self):
        return self.len


    # 未划分数据集
    # def read_label_list(self, label_dir, image_dir, mode, seed):
    #     labels_seperated = []
    #     path = '/home/lokfar/longys/web-data/results-15-16_china'
    #     # folds = sorted(os.listdir(path))
    #     # for fold in folds:
    #     #     pred = os.path.join(path, fold)
    #     #
    #     #     labels_seperated.append(str(pred))
    #
    #     folds = sorted(os.listdir(path))
    #     for fold in folds:
    #         pred = os.path.join(path, fold, 'pred')
    #         for name in os.listdir(pred):
    #             name_path = os.path.join(pred, name)
    #             # print(name_path)
    #             # print(type(name_path))
    #             labels_seperated.append(str(name_path))
    #
    #     if self.mode != 'test':
    #         np.random.seed(seed)
    #         np.random.shuffle(labels_seperated)
    #
    #     return labels_seperated


    def read_label_list(self, label_dir, image_dir, mode, seed):
        # labels_seperated = []
        # path = '/home/lokfar/longys/web-data/results-15-16_china_202006_8'
        # # folds = sorted(os.listdir(path))
        # # for fold in folds:
        # #     pred = os.path.join(path, fold)
        # #
        # #     labels_seperated.append(str(pred))
        #
        # times = sorted(os.listdir(path))
        #
        # for time in times:
        #     img_fold=os.listdir(os.path.join(path,time,'pred'))
        #     for img_name in img_fold:
        #         img_path=os.path.join(path,time,'pred',img_name)
        #         print(img_path)
        #         labels_seperated.append(img_path)

        labels_seperated = []
        path = '/mnt/B/daikuai/satellite/satellite/results-15-16_china_202006_24'
        times = sorted(os.listdir(path))
        for time in times:
            img_fold=os.listdir(os.path.join(path,time))
            for img_name in img_fold:
                img_path=os.path.join(path,time,img_name)
                print(img_path)
                labels_seperated.append(img_path)

        if self.mode != 'test':
            np.random.seed(seed)
            np.random.shuffle(labels_seperated)

        return labels_seperated

    def load_data(self, image_dir, label_dir, name, mode):
        '''
        加载数据
        :return:
        '''
        # print(os.path.join(image_dir, name[:6], name))
        # print(os.path.join(label_dir, mode, name[:6], name))
        # path = '/home/lokfar/longys/web-data/results-15-16_south_east_202006'
        # image = cv2.imread(os.path.join(path, name[:6], name), cv2.IMREAD_GRAYSCALE)

        image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(name)
        print(image)
        # 东南亚参数
        # new_image = np.zeros((1504, 1760), dtype=int)
        # new_image[2:-2, 5:-5] = image
        #
        # r, c = new_image.shape
        # # print("r  c", r,c)
        # image = np.reshape(new_image, (1, r, c)) / 255.0
        # labels = np.zeros((1, r, c))

        #中国区参数
        # new_image = np.zeros((768, 1280), dtype=int)
        # new_image[19:749, :] = image
        #
        # r, c = new_image.shape
        # image = np.reshape(new_image, (1, r, c)) / 255.0
        # labels = np.zeros((1, r, c))

        #扩大后中国区参数
        new_image = np.zeros((1088, 1280), dtype=int)
        new_image[14:1074, :] = image

        r, c = new_image.shape
        image = np.reshape(new_image, (1, r, c)) / 255.0
        labels = np.zeros((1, r, c))

        return image.copy(), np.array(labels, dtype=int)

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        # data = self.toTensor(data)
        data = torch.Tensor(data)
        return data

# if __name__ == "__main__":
#     image_dir = '/extend/shixc/bright_images/'
#     label_dir = '/extend/shixc/labels_v2/'
#     # time_start = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
#
#     # checkpoint_path = '/extend/shixc/data_1/unet_0701/checkpoints/{}/'.format \
#     #     (time_start)
#     # os.makedirs(checkpoint_path)
#     path = '/extend/shixc/data_1/deepv3_0706/checkpoints/png/'
#     # label_path = '/extend/shixc/data_1/labels_v2/all/201806/20180611214500_20180611214916.png'
#     # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#     # label = label[:, 440:1240]
#     # label[0:192, 0:355] = 0
#     # _edgemap = edge_utils.mask_to_onehot(label + 1, 2)
#     # _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
#     # print(_edgemap[0])
#     # cv2.imwrite('/extend/shixc/data_1/a.png', _edgemap[0] * 255)
#     train_data = UnetDataset(label_dir=label_dir, image_dir=image_dir, mode='train')
#     train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=8)
#     #
#     val_data = UnetDataset(label_dir=label_dir, image_dir=image_dir, mode='val')
#     val_loader = DataLoader(dataset=val_data, batch_size=2, shuffle=False, num_workers=4)
#     #
#     test_data = UnetDataset(label_dir=label_dir, image_dir=image_dir, mode='test')
#     test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=False, num_workers=4)
#     for step, (x, b_label, name) in enumerate(train_loader):
#         # pass
#         print("step is {}, x.shape is {}, label.shape is {}".format(step, x.shape, b_label.shape))
#         print(torch.max(b_label[:, 1]), torch.min(b_label[:, 1]))

if __name__ == "__main__":
    test_data = DugsDataset(label_dir=LABEL_DIR, image_dir=IMAGE_DIR, mode='test')
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=4,
                             drop_last=False)
    save_path = '/home/lokfar/lushenyuan/pythonProject/20220707/test_save'
    for step, (x, b_label, name) in enumerate(test_loader):
        print("name is {}".format(name))
        for b_i in range(len(name)):
            spilts=name[b_i].split('/')
            time=spilts[-2]
            if not os.path.exists(os.path.join(save_path,time)):
                os.makedirs(os.path.join(save_path,time))
            spilts[-1]=spilts[-1].replace('jpg','png')
            print(name[b_i],os.path.join(save_path,time,spilts[-1]))