import numpy as np
import os
import time
import sys
import json

log_path = '/extend/shixc/section_one/signle/all/tranditional/220/log_test.txt'
print(log_path)
# 按年份统计指标，是根据不同类别的像素数目计算，不根据指标的直接求均值计算
file = open(log_path, 'r')
line = file.readline()
line = file.readline()
data = dict()
while line:
    contexts = line.split(",")
    # for con in contexts:
    #     print(con)
    if contexts[0] not in data.keys():
        data[contexts[0]] = []
        data[contexts[0]].append(contexts[1].lstrip())
        data[contexts[0]].append(contexts[2].lstrip())
        data[contexts[0]].append(contexts[3].lstrip())
        data[contexts[0]].append(contexts[8].replace("\n", ""))
    # else:
    #     if contexts[8].replace("\n", "") > data[contexts[0]][3]:
    #         data[contexts[0]][0] = contexts[1].lstrip()
    #         data[contexts[0]][1] = contexts[2].lstrip()
    #         data[contexts[0]][2] = contexts[3].lstrip()
    #         data[contexts[0]][3] = contexts[8].replace("\n", "")


    line = file.readline()
file.close()

print(data)
# print(len(data))
all_data = [0.0, 0.0, 0.0]
for key in data.keys():
    for i in range(3):
        all_data[i] += int(data[key][i])

print(all_data)


pod = all_data[2] / all_data[0]
far = (all_data[1] - all_data[2]) / all_data[1]
csi = all_data[2] / (all_data[0] + all_data[1] - all_data[2])
f1 = (2 * all_data[2]) /(all_data[0] + all_data[1])
print("year: {}, avg_pod: {}, avg_far: {}, avg_csi: {}, avg_f1: {}".format(2018, "%.4f"%pod, "%.4f"%far, "%.4f"%csi, "%.4f"%f1))