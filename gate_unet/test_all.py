# -*- coding: utf-8 -*-
from  PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboard_logger import Logger
# from unet_model import UNet
from gated_s_unet import unet_seg, multi_loss_layer
import os
import cv2
import numpy as np
import time
from my_exception import *
from config_gated import *
import torchvision.transforms as transforms

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

tenb_logger = Logger(logdir="./tensorboard_logs", flush_secs=10)
checkpoint_path = './checkpoints/{}/'.format \
    (time_start)
# checkpoint_path = '/extend/RemoteSensingProject/ResultData/FY3B_VIRR_North_Africa_DL/result/mst_disturb_label_vggm_avg_pool_single_channel_t_20191031_184354_sz_512_lr_5e-05_bs_64_ch_2/'
logs_path = checkpoint_path + 'logs.txt'

# train_data = UnetDataset(label_dir=LABEL_DIR, image_dir=IMAGE_DIR, mode='train')
# train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
#
# val_data = UnetDataset(label_dir=LABEL_DIR, image_dir=IMAGE_DIR, mode='val')
# val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=4)

test_data = DugsDataset(label_dir=LABEL_DIR, image_dir=IMAGE_DIR, mode='test')
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=8, drop_last=False)


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
if MODE == 'train':
    for epoch in range(EPOCH):

        # scheduler.step()
        model.train()
        losses_tr = []
        for step, (x, b_label, name) in enumerate(train_loader):
            # print(x.shape)

            b_x = x.cuda()  # batch x, shape (batch, 28*28)
            b_y = b_label.cuda()  # batch y, shape (batch, 28*28)
            pred, edge = model(b_x)

            # print(type(pred))

            loss_ce, loss_edge, loss_att, loss_dual = (0.0, 0.0, 0.0, 0.0)
            # l_ce,
            l_ce, l_edge, l_att, l_dual = joint_loss_train((pred.float(), edge.float()),
                                                     (b_y[:, 0] - 1, b_y[:, 1].float()))
            loss_ce += l_ce.mean()
            loss_edge += l_edge.mean()
            loss_att += l_att.mean()
            loss_dual += l_dual.mean()

            # print(pred.shape, b_y.shape)
            pred = pred.permute(0, 2, 3, 1)
            m, width_out, height_out = pred.shape[:3]
            # Resizing the outputs and label to caculate pixel wise softmax loss
            pred = pred.resize(m * width_out * height_out, CHANNEL_OUT)
            seg_true = b_y[:, 0]
            seg_true = seg_true.resize(m * width_out * height_out)
            # b_y = b_y.resize(m * width_out * height_out)
            loss_seg = 0.0
            loss_seg = seg_loss_func(pred, seg_true-1)
            # + loss_att + loss_dual + loss_ce
            loss = loss_seg * 20.0 + loss_edge * 20.0 + loss_att + loss_dual + loss_ce * 20.0
            # loss = loss_layer([loss_seg, loss_edge, loss_att, loss_dual, loss_ce])
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # 考虑连通区域是否命中加入loss，即是否检测出对流
            losses_tr.append(loss_seg.detach().cpu().numpy())
            if losses_tr[-1] is np.nan:
                print(name)
                raise LossNanError
            tenb_logger.log_value('train_loss_total', loss.item(), epoch * len(train_loader) + step)
            tenb_logger.log_value('train_loss_seg', loss_seg.item(), epoch * len(train_loader) + step)
            tenb_logger.log_value('train_loss_edge', loss_edge.item(), epoch * len(train_loader) + step)
            print('Epoch {}, Step {}, total_loss: {}, seg_loss: {}, edge_loss: {}'.format(epoch + 1, step, loss.detach().cpu().numpy(), losses_tr[-1], loss_edge.detach().cpu().numpy()))
            # if step > 2:
            #     break

        model.eval()
        losses_val = []
        miou = []
        metric_dict = {'pa': pa_metric, 'pod': pod_metric, 'far': far_metric, 'csi': csi_metric}
        result_dict = {'pa': [], 'pod': [], 'far': [], 'csi': []}
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        if epoch % SAVE_PERIOD == 0:
            save_path = os.path.join(checkpoint_path, 'val' + str(epoch + 1))
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        with torch.no_grad():
            for step, (x, b_label, name) in enumerate(val_loader):
                b_x = x.cuda()  # batch x, shape (batch, 28*28)
                b_y = b_label.cuda()  # batch y, shape (batch, 28*28)

                pred, edge = model(b_x)

                loss_ce, loss_edge, loss_att, loss_dual = (0.0, 0.0, 0.0, 0.0)
                # l_ce,
                l_ce, l_edge, l_att, l_dual = joint_loss_train((pred.float(), edge.float()),
                                                         (b_y[:, 0] - 1, b_y[:, 1].float()))
                loss_ce += l_ce.mean()
                loss_edge += l_edge.mean()
                loss_att += l_att.mean()
                loss_dual += l_dual.mean()


                pred = pred.permute(0, 2, 3, 1)
                m, width_out, height_out = pred.shape[:3]
                # Resizing the outputs and label to caculate pixel wise softmax loss
                pred_resize = pred.resize(m * width_out * height_out, CHANNEL_OUT)
                seg_true = b_y[:, 0]
                seg_true = seg_true.resize(m * width_out * height_out)
                # b_y = b_y.resize(m * width_out * height_out)

                
                losses_val.append(np.array(seg_loss_func(pred_resize, seg_true - 1).cpu()))
                pred = torch.argmax(pred, dim=3)

                for i_b in range(len(pred)):
                    if (b_y[i_b, 0] - 1).sum() < LABEL_THRESHOLD:
                        pred[i_b] = torch.Tensor(np.zeros(pred[i_b].size())).cuda()
                        b_y[i_b, 0] = torch.Tensor(np.ones(b_y[i_b, 0].size())).cuda()

                miou.append(iou_metric(pred.float(), (b_y[:, 0] - 1).squeeze(1).float()).mean())
                tenb_logger.log_value('val_loss_seg', losses_val[-1], epoch * len(val_loader) + step)
                tenb_logger.log_value('val_loss_edge', loss_edge.item(), epoch * len(val_loader) + step)
                tenb_logger.log_value('val_miou', miou[-1], epoch * len(val_loader) + step)
                print('Epoch {}, Step {}, loss_seg: {}, loss_edge: {}, miou: {}'.format(epoch + 1, step, losses_val[-1], loss_edge.detach().cpu().numpy(), miou[-1]))
                for key in metric_dict:
                    score = metric_dict[key](pred.float(), (b_y[:, 0] - 1).squeeze(1).float())
                    result_dict[key].append(score.mean())

                # 保存分割结果，真实图像和label
                pred = pred.cpu().numpy()
                x = x[:, 0, :, :].squeeze(1).cpu().numpy()
                b_y = b_y[:, 0, :, :].squeeze(1).cpu().numpy()
                if epoch % SAVE_PERIOD == 0:
                    for b_i in range(len(pred)):
                        # print(pred[b_i].sum(), b_y[b_i] .sum())
                        if not os.path.exists(os.path.join(save_path, name[b_i][:6])):
                            os.mkdir(os.path.join(save_path, name[b_i][:6]))
                        # if not os.path.exists(os.path.join(save_path, name[b_i][:6], name[b_i][:29])):
                        #     os.mkdir(os.path.join(save_path, name[b_i][:6], name[b_i][:29]))


                        img_labeled = merge_img(x[b_i]*255, b_y[b_i] - 1, pred[b_i])
                        cv2.imwrite(os.path.join(save_path, name[b_i][:6], name[b_i]), img_labeled)
                        # cv2.imwrite(os.path.join(save_path, name[b_i][:6], name[b_i][-5]+'.png'), img_labeled)

                # if step > 2:
                #     break

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        if not os.path.exists(logs_path):
            log_file = open(logs_path, 'w')
            log_file.write('epoch,loss_tr,loss_val,miou,' + ','.join(metric_dict.keys()) + ',time\n')
        else:
            log_file = open(logs_path, 'a')
        time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        # log_file.write('{},{},{},{},{}\n'.format(epoch+1, np.mean(losses_tr), np.mean(losses_val), np.mean(miou), time_now))
        record = str(epoch + 1) + ',%.6f' % (np.mean(losses_tr)) + ',%.6f' % (np.mean(losses_val)) + ',%.4f' % (
            np.mean(miou)) + ',' + \
                 ','.join(['%.4f' % (np.mean(result_dict[key])) for key in metric_dict]) + ',' + time_now + '\n'
        log_file.write(record)
        log_file.close()

        # update last snapshot
        if os.path.exists(last_snapshot) and epoch < 16:
            os.remove(last_snapshot)
        last_snapshot = checkpoint_path + 'last-m-' + str(epoch + 1) + '-' + str("%.4f" % np.mean(losses_tr)) \
                        + '-' + str("%.4f" % np.mean(losses_val)) \
                        + '-' + str("%.4f" % np.mean(miou)) + '.pth.tar'
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': np.mean(losses_val),
                    'optimizer': optimizer.state_dict()}, last_snapshot)

        # update best snapshot
        is_update = False
        if miou_max < np.mean(miou):
            miou_max = np.mean(miou)
            is_update = True
        if loss_val_min > np.mean(losses_val):
            loss_val_min = np.mean(losses_val)
            is_update = True
        if is_update:
            # if os.path.exists(best_snapshot):
            #     os.remove(best_snapshot)
            best_snapshot = checkpoint_path + 'best-m-' + str(epoch + 1) + '-' + str("%.4f" % np.mean(losses_tr)) \
                            + '-' + str("%.4f" % np.mean(losses_val)) \
                            + '-' + str("%.4f" % np.mean(miou)) + '.pth.tar'
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': np.mean(losses_val),
                        'optimizer': optimizer.state_dict()}, best_snapshot)


elif MODE == 'test':
    model.eval()
    losses_val = []
    miou = []
    metric_dict = {'pa': pa_metric, 'pod': pod_metric, 'far': far_metric, 'csi': csi_metric}
    result_dict = {'pa': [], 'pod': [], 'far': [], 'csi': []}
    save_path = './test_save'
    # save_path = '/home/lokfar/longys/web-data/mask/china'
    #save_path = '/home/lokfar/longys/web-data/mask/south_east'
    # save_path = '/extend/shixc/data_1/unet_0716/tensorboard_logs/new_test/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        for step, (x, b_label, name) in enumerate(test_loader):
            print("正在测试")
            b_x = x.cuda()  # batch x, shape (batch, 28*28)
            b_y = b_label.cuda()  # batch y, shape (batch, 28*28)

            pred, edge = model(b_x)
            pred = pred.permute(0, 2, 3, 1)
            pred = torch.argmax(pred, dim=3)
            # 保存分割结果，真实图像和label
            pred = pred.cpu().numpy()
            print("Step: {}".format(step))
            #pred = pred[:, 19:749, :]  #中国区参数
            # print("pred shape",pred.shape)
            #pred = pred[:,2:-2, 5:-5]  #东南亚参数
            pred = pred[:, 14:1074, :]  # 中国区扩大后参数
            for b_i in range(len(pred)):

                # name_split = name[b_i].split("/")
                #
                # img_save_path = os.path.join(name_split[1], name_split[2], name_split[3], name_split[4],
                #                              name_split[5], name_split[6], "pred_mark")
                # img_save_path = "/" + img_save_path
                # if not os.path.exists(img_save_path):
                #     os.mkdir(img_save_path)


                spilts = name[b_i].split('/')
                timestring = spilts[-2]
                if not os.path.exists(os.path.join(save_path, timestring)):
                    os.makedirs(os.path.join(save_path, timestring))
                spilts[-1] = spilts[-1].replace('jpg', 'png')
                img_save_path = os.path.join(save_path,timestring,spilts[-1])

                # print(pred.shape)
                if SAVE_IMAGE == 1:
                    black_img = pred[b_i] * 255

                    r, c = black_img.shape
                    # print("r  c",r,c)

                    new_img = np.zeros((r, c, 4), dtype=np.int8)

                    index = black_img == 255

                    new_img[:, :, 0] = black_img
                    new_img[:, :, 1] = black_img
                    new_img[:, :, 2] = black_img
                    new_img[:, :, -1] = np.where(black_img > 0, 255, 0)


                    new_img[index, 0] = 67
                    new_img[index, 1] = 205
                    new_img[index, 2] = 128

                    img = Image.fromarray(new_img, mode="RGBA")

                    img.save(img_save_path)
                    print(img_save_path)




                # 对真实图片检测的逻辑
                # name_split = name[b_i].split("/")
                #
                # img_save_path = os.path.join(save_path,"202006_mark")
                # # img_save_path = "/"+img_save_path
                # if not os.path.exists(img_save_path):
                #     os.mkdir(img_save_path)
                # # print(pred.shape)
                # if SAVE_IMAGE == 1:
                #     cv2.imwrite(os.path.join(img_save_path, name_split[-1]), pred[b_i] * 255)
                #
                #     # black_img = pred[b_i]*255
                #     #
                #     # r , c = black_img.shape
                #     # # print("r  c",r,c)
                #     #
                #     new_img = np.zeros((r,c,4),dtype=np.int8)
                #
                #     index = black_img == 255
                #
                #     new_img[:, :, 0] = black_img
                #     new_img[:, :, 1] = black_img
                #     new_img[:, :, 2] = black_img
                #     new_img[:, :, -1] = np.where(black_img > 0, 255, 0)
                #
                #     new_img[index, 0] = 67
                #     new_img[index, 1] = 205
                #     new_img[index, 2] = 128
                #     #
                #     #
                #     img = Image.fromarray(new_img, mode="RGBA")
                #
                #     img.save(os.path.join(img_save_path, name_split[-1]))

                    # name_split = name[b_i].split("/")
                    #
                    # img_save_path = os.path.join(name_split[1], name_split[2], name_split[3], name_split[4],
                    #                              name_split[5], name_split[6], "pred_mark")
                    # img_save_path = "/" + img_save_path
                    # if not os.path.exists(img_save_path):
                    #     os.mkdir(img_save_path)
                    # # print(pred.shape)
                    # if SAVE_IMAGE == 1:
                    #     black_img = pred[b_i] * 255
                    #
                    #     r, c = black_img.shape
                    #     # print("r  c",r,c)
                    #
                    #     new_img = np.zeros((r, c, 4), dtype=np.int8)
                    #
                    #     index = black_img == 255
                    #
                    #     new_img[:, :, 0] = black_img
                    #     new_img[:, :, 1] = black_img
                    #     new_img[:, :, 2] = black_img
                    #     new_img[:, :, -1] = np.where(black_img > 0, 255, 0)
                    #
                    #     new_img[index, 0] = 67
                    #     new_img[index, 1] = 205
                    #     new_img[index, 2] = 128
                    #
                    #     img = Image.fromarray(new_img, mode="RGBA")
                    #
                    #     img.save(os.path.join(img_save_path, name_split[-1]))

                    # cv2.imwrite(os.path.join(img_save_path, name_split[-1]), pred[b_i]*255)

            # 中国区参数
            # for b_i in range(len(pred)):
            #     if not os.path.exists(os.path.join(save_path, name[b_i][:6])):
            #         os.mkdir(os.path.join(save_path, name[b_i][:6]))
            #     # print(pred.shape)
            #     if SAVE_IMAGE == 1:
            #         cv2.imwrite(os.path.join(save_path, name[b_i][:6], name[b_i][:-15]+'.png'), pred[b_i]*255)
