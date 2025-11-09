# import os
# import time
# import json
#
# train_path = '/extend/shixc/labels_v2/all/sequences_4_single_new_train.json'
# val_path = '/extend/shixc/labels_v2/all/sequences_4_single_new_val.json'
# test_path = '/extend/shixc/labels_v2/all/sequences_4_single_new_test.json'
#
# train = json.load(open(train_path, 'r'))
# val = json.load(open(val_path, 'r'))
# test = json.load(open(test_path, 'r'))
#
# train = set(train)
# val = set(val)
# test = set(test)
#
# train_test = train.intersection(test)
# val_test = val.intersection(test)
# train_val = train.intersection(val)
# print(train_test)
# print(len(train_test))
# print(val_test)
# print(len(val_test))
# print(train_val)
# print(len(train_val))