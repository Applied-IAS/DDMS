import os.path as osp
import time
import cv2
import numpy as np
import edge_utils
path = '/extend/shixc/labels_v2/all/201807/20180715081500_20180715081916.png'
img_path = '/extend/shixc/bright_images/201807/20180715081500_20180715081916_bright_img.png'
label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
label = 1 * (label >= 200)
label = label[:, 440:1240]
image = image[:, 440:1240]
label[0:192, 0:355] = 0
_edgemap = edge_utils.mask_to_onehot(label + 1, 2)
_edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
edge_mask = _edgemap[0]
edge_mask = edge_mask * 255
label = label * 255
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180715081500edge_mask.png', edge_mask)
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180715081500image.png', image)
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180715081500label.png', label)

path = '/extend/shixc/labels_v2/all/201807/20180715134500_20180715134916.png'
img_path = '/extend/shixc/bright_images/201807/20180715134500_20180715134916_bright_img.png'
label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
label = 1 * (label >= 200)
label = label[:, 440:1240]
image = image[:, 440:1240]
label[0:192, 0:355] = 0
_edgemap = edge_utils.mask_to_onehot(label + 1, 2)
_edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
edge_mask = _edgemap[0]
edge_mask = edge_mask * 255
label = label * 255
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180715134500edge_mask.png', edge_mask)
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180715134500image.png', image)
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180715134500label.png', label)

path = '/extend/shixc/labels_v2/all/201808/20180814074500_20180814074916.png'
img_path = '/extend/shixc/bright_images/201808/20180814074500_20180814074916_bright_img.png'
label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
label = 1 * (label >= 200)
label = label[:, 440:1240]
image = image[:, 440:1240]
label[0:192, 0:355] = 0
_edgemap = edge_utils.mask_to_onehot(label + 1, 2)
_edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
edge_mask = _edgemap[0]
edge_mask = edge_mask * 255
label = label * 255
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180814074500edge_mask.png', edge_mask)
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180814074500image.png', image)
cv2.imwrite('/extend/shixc/lunwen/d_gated_u_net_all_area/edge_pred_result/20180814074500label.png', label)

