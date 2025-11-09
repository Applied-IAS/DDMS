import numpy as np
import cv2



iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def iou(img_true, img_pred):
    # img_pred = (img_pred > 0).float()
    i = (img_true * img_pred).sum()
    u = ((img_true + img_pred) > 0).float().sum()
    return i / u if u != 0 else u

# 不同类别IoU的均值
def iou_metric(imgs_pred, imgs_true, num_classes=2):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    # batch
    for i in range(num_images):
        c_score = 0.0
        # classes
        for c in range(num_classes):
            c_true = (imgs_true[i] == c).float()
            c_pred = (imgs_pred[i] == c).float()
            if c_true.sum() == c_pred.sum() == 0:
                c_score += 1
            else:
                # scores[i] = (iou_thresholds <= iou(imgs_true[i], imgs_pred[i]).cpu().numpy()).mean()
                c_score += iou(c_true, c_pred).cpu().numpy()
        scores[i] = c_score / num_classes
    return scores

# 正确预测像素数/全部像素数
def pa_metric(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        scores[i] = ((imgs_pred[i] == imgs_true[i]).float().sum() / imgs_true[i].numel()).cpu().numpy()
    return scores


# pod 命中率 正确预测的正样本(TP) / (正确预测的正样本(TP) + 漏检的正样本(FN))即label中的所有正样本
def pod_metric(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        label_positive = imgs_true[i].sum()
        pred_positive = imgs_pred[i].sum()
        if label_positive == 0:
            if pred_positive == 0:
                scores[i] = 1
            else:
                scores[i] = 0
        else:
            scores[i] = ((imgs_pred[i] * imgs_true[i]).float().sum() / label_positive).cpu().numpy()
    return scores

# far 误报率 错误预测的正样本(FP) / (正确预测的正样本(TP) + 错误预测的正样本(FP))即预测值中的所有正样本
def far_metric(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        pred_positive = imgs_pred[i].sum()
        if pred_positive == 0:
            scores[i] = 0
        else:
            scores[i] = ((imgs_pred[i]*((imgs_true[i]==0).float())).sum() / pred_positive).cpu().numpy()
    return scores

# csi 正确预测的正样本(TP) / (正确预测的正样本(TP) + 漏检的正样本(FN) + 错误预测的正样本(FP))即预测和真实正样本的并集
# 就是正样本的IoU
def csi_metric(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        positive = ((imgs_pred[i] + imgs_true[i])>0).sum()
        if positive == 0:
            scores[i] = 1
        else:
            scores[i] = ((imgs_pred[i] * imgs_true[i]).float().sum() / positive).cpu().numpy()
    return scores

# def iou(img_true, img_pred):
#     img_pred = (img_pred > 0)
#     i = (img_true * img_pred).sum()
#     u = ((img_true + img_pred) > 0).sum()
#     return i / u if u != 0 else u

# def iou_metric(imgs_pred, imgs_true):
#     if imgs_true.sum() == imgs_pred.sum() == 0:
#         scores = 1
#     else:
#         scores = iou(imgs_true, imgs_pred)
#     return scores

def merge_img(img, img_label, img_pred):
    # 漏检为蓝色 预测正确为绿色 预测错误为红色  cv2 BGR
    # r, c = img.shape
    # img_labeled = np.ones((r, c, 3)) * 255
    # labeled = np.zeros((r, c))
    # # 漏检为蓝色
    # img_labeled[:, :, 0] = img
    # idx = (img_label > 0) & (img_pred == 0)
    # img_labeled[idx, 0] = 255
    # labeled[idx] = 180
    # # 预测正确为绿色
    # img_labeled[:, :, 1] = img
    # idx = (img_label > 0) & (img_pred > 0)
    # img_labeled[idx, 1] = 255
    # labeled[idx] = 255
    # # 预测错误为红色
    # img_labeled[:, :, 2] = img
    # idx = (img_label == 0) & (img_pred > 0)
    # img_labeled[idx, 2] = 255
    # labeled[idx] = 200
    r, c = img.shape
    img_labeled = np.ones((r, c, 3)) * 255
    img_labeled[:, :, 0] = img.copy()
    img_labeled[:, :, 1] = img.copy()
    img_labeled[:, :, 2] = img.copy()
    labeled = np.zeros((r, c))
    # 漏检为蓝色
    idx = (img_label > 0) & (img_pred == 0)
    img_labeled[idx, 0] = 255
    img_labeled[idx, 1] = 144
    img_labeled[idx, 2] = 30
    labeled[idx] = 180
    # 预测正确为绿色
    idx = (img_label > 0) & (img_pred > 0)
    img_labeled[idx, 1] = 255
    img_labeled[idx, 0] = 127
    img_labeled[idx, 2] = 0
    labeled[idx] = 255
    # 预测错误为红色
    idx = (img_label == 0) & (img_pred > 0)
    img_labeled[idx, 2] = 220
    img_labeled[idx, 1] = 20
    img_labeled[idx, 0] = 60
    labeled[idx] = 200

    return img_labeled
    # return labeled

if __name__ == "__main__":

    # date = '20180214010000_20180214011459'
    # pkg = '12-1'
    # # date = '20180620201500_20180620201916'
    # # pkg = '12-4'
    # # date = '20180815093000_20180815093416'
    # # pkg = '12-5'
    # img_pred = cv2.imread('../pred/'+date+'.png', cv2.IMREAD_GRAYSCALE)
    # img_label = cv2.imread('../../label_convection/resources/labeled_data/labels/val/'+date[:6]+'/'+date+'.png', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('../../label_convection/resources/labeled_data/all/'+pkg+'/'+date+'/bright_img_ori.png', cv2.IMREAD_GRAYSCALE)
    # print(iou_metric(np.array(img_pred, dtype=float), np.array(img_label[40:-40, 20:-20] > 0, dtype=float)))
    #
    # img = img[40:-40, 20:-20]
    # img_label = img_label[40:-40, 20:-20]
    # img_labeled = merge_img(img, img_label, img_pred)
    #
    # cv2.imwrite('../pred/'+date+'_merged.png', img_labeled)
    import os
    import cv2
    path = '/extend/shixc/page/'
    # print(os.listdir(path))

    img = cv2.imread(os.path.join(path, '1.png'), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(os.path.join(path, '1l.png'), cv2.IMREAD_GRAYSCALE)
    # label = 1 * (label > 0)

    r, c = img.shape
    img_labeled = np.ones((r, c, 3)) * 255
    img_labeled[:, :, 0] = img.copy()
    img_labeled[:, :, 1] = img.copy()
    img_labeled[:, :, 2] = img.copy()

    # 预测正确为绿色
    idx = (label > 0)
    img_labeled[idx, 1] = 255
    img_labeled[idx, 0] = 127
    img_labeled[idx, 2] = 0

    cv2.imwrite(os.path.join(path, 'a.png'), img_labeled)

    img = cv2.imread(os.path.join(path, '2.png'), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(os.path.join(path, '2l.png'), cv2.IMREAD_GRAYSCALE)
    # label = 1 * (label > 0)

    r, c = img.shape
    img_labeled = np.ones((r, c, 3)) * 255
    img_labeled[:, :, 0] = img.copy()
    img_labeled[:, :, 1] = img.copy()
    img_labeled[:, :, 2] = img.copy()

    # 预测正确为绿色
    idx = (label > 0)
    img_labeled[idx, 1] = 255
    img_labeled[idx, 0] = 127
    img_labeled[idx, 2] = 0

    cv2.imwrite(os.path.join(path, 'b.png'), img_labeled)

    img = cv2.imread(os.path.join(path, '3.png'), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(os.path.join(path, '3l.png'), cv2.IMREAD_GRAYSCALE)
    # label = 1 * (label > 0)

    r, c = img.shape
    img_labeled = np.ones((r, c, 3)) * 255
    img_labeled[:, :, 0] = img.copy()
    img_labeled[:, :, 1] = img.copy()
    img_labeled[:, :, 2] = img.copy()

    # 预测正确为绿色
    idx = (label > 0)
    img_labeled[idx, 1] = 255
    img_labeled[idx, 0] = 127
    img_labeled[idx, 2] = 0

    cv2.imwrite(os.path.join(path, 'c.png'), img_labeled)

    img = cv2.imread(os.path.join(path, '4.png'), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(os.path.join(path, '4l.png'), cv2.IMREAD_GRAYSCALE)
    # label = 1 * (label > 0)

    r, c = img.shape
    img_labeled = np.ones((r, c, 3)) * 255
    img_labeled[:, :, 0] = img.copy()
    img_labeled[:, :, 1] = img.copy()
    img_labeled[:, :, 2] = img.copy()

    # 预测正确为绿色
    idx = (label > 0)
    img_labeled[idx, 1] = 255
    img_labeled[idx, 0] = 127
    img_labeled[idx, 2] = 0

    cv2.imwrite(os.path.join(path, 'd.png'), img_labeled)
