import os
import numpy as np
# path = '/home/lokfar/longys/web-data/results-15-16_south_east_202006'
# folds = sorted(os.listdir(path))
# for fold in folds:
#     pred = os.path.join(path, fold,'pred')
#     for name in sorted(os.listdir(pred)):
#         name_path = os.path.join(pred,name)
#         print(name_path)
#
#     break
import cv2
from  PIL import Image
name = '/home/lokfar/longys/web-data/ori_data/southeast/202006/20200601000000_20200601001459_orl_img.png'
# image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
# print(image.shape)

name_split = name.split("/")
print(name_split)
img_save_path = os.path.join(name_split[1],name_split[2],name_split[3],name_split[4],name_split[5],name_split[6],"pred_mark")
print(img_save_path)
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)


# out_image = cv2.imread('/home/lokfar/longys/web-data/mask/south_east/202006/20200601000000_20200601001459_bright_img.png')
#
# print("out_img", out_image.shape)
# r,c = out_image[:,:,0].shape
# new_img = np.zeros((r,c,4), dtype=np.int8)
# print("new_img" , new_img.shape)
# index = out_image[:,:,0] == 255
#
# black_img = out_image[:,:,0]
# print(black_img.shape)
#
# new_img[:,:,0] = black_img
# new_img[:,:,1] = black_img
# new_img[:,:,2] = black_img
# new_img[:,:,-1] = np.where(black_img>0,255,0)
#
# # 67 205 128
# # print(out_image[:,:,0].shape)
# #
# #
# # new_image = np.zeros((r, c,3), dtype=int)
# # print(new_image.shape)
#
# new_img[index,0] = 67
# new_img[index,1] = 205
# new_img[index,2] = 128
#
# # cv2.imwrite("color_marks.png",new_img)
#
# img = Image.fromarray(new_img, mode="RGBA")
#
# img.save("color_marks.png")

