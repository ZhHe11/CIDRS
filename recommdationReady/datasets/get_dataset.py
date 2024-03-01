import pandas as pd
#
filename = "/home/data/zh/project1/streamlit/recommadation/datasets/img_id.csv"
img_ids = pd.read_csv(filename)

img_ids.to_csv('./imgs_large.csv', index=False)

# print(len(img_ids))
# # split 7:3
train_img_ids = img_ids[:int(len(img_ids)*0.1)]
train_large = img_ids[:int(len(img_ids)*0.5)]
val = img_ids[int(len(img_ids)*0.8):int(len(img_ids)*0.9)]
test_img_ids = img_ids[int(len(img_ids)*0.9):]
print(len(train_img_ids))
print(len(train_large))
print(len(val))
print(len(test_img_ids))
print(len(img_ids[:int(len(img_ids)*0.8)]))
print(len(img_ids))

# # save as csv file
# train_img_ids.to_csv("/home/data/zh/project1/streamlit/recommadation/datasets/train_img_id1.csv", index=False)
# train_large.to_csv("./train_large.csv", index=False)
# val.to_csv("/home/data/zh/project1/streamlit/recommadation/datasets/val.csv", index=False)
# test_img_ids.to_csv("/home/data/zh/project1/streamlit/recommadation/datasets/test_img_id.csv", index=False)

# objs_file = pd.read_csv("/hy-nas/zhanghe/data/fur/txt/n_dict-less-3.0.csv", encoding='utf-8', header=None)
# print(objs_file[0].values)

# text = clip.tokenize(objs).to(device)
#

# import numpy as np
# success_trun = [1, 0, 1, 0, 2]
# success_trun_all = [x for x in success_trun if x != 0]
# a = []
# a = a + success_trun_all
# print(a)
