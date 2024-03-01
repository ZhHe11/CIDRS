import os
import re
import cv2
import sys
import json

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import matplotlib.image as mping
import torchvision.transforms as transforms

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from Model.CLIP.cn_clip.clip import load_from_name
import copy


device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_clip_model2 = '005/epoch_50.pt'
# resume = '/hy-nas/zhanghe/project/CLIP/Chinese-CLIP-master/Chinese-CLIP-master/usage/save_path' + '/' + sentence_clip_model2
resume = '/home/data/zh/project1/SaveModel/SentenceEpoch_50.pt'
sentence_clip_model2, sentence_clip_preprocess2 = load_from_name("ViT-B-16", device=device, download_root='../../data/pretrained_weights/',
                                   resume=resume)


class Image_preprocess(data.Dataset):
    def __init__(self, img_ids_filename):
        img_ids = pd.read_csv(img_ids_filename)
        self.img_ids = img_ids['img_id'].values
        # self.sentence_clip_preprocess2 = sentence_clip_preprocess2
    def preprocess(self, img_id):
        img = Image.open("/home/data/zh/fur/processed_img" + '/' + str(img_id) + '.jpg')
        img = sentence_clip_preprocess2(img)
        return img

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = self.preprocess(img_id).float()
        return img, img_id

    def __len__(self):
        return len(self.img_ids)


class MemoryLoaderDataset(data.Dataset):
    def __init__(self, device):
        self.memory = []
        self.memory_size = 0
        self.device = device


    def store_mem(self, g, f, c, hx, a, r, g_, f_, c_, hx_, batch_size):
        for i in range(batch_size):
            if r[i] <= -1000:
                continue
            else:
                g_tmp = copy.deepcopy(g[i])
                f_tmp = copy.deepcopy(f[i])
                c_tmp = copy.deepcopy(c[i])
                hx_tmp = copy.deepcopy(hx[i])
                a_tmp = copy.deepcopy(a[i])
                reward_temp = copy.deepcopy(r[i])
                g_tmp_ = copy.deepcopy(g_[i])
                f_tmp_ = copy.deepcopy(f_[i])
                c_tmp_ = copy.deepcopy(c_[i])
                hx_tmp_ = copy.deepcopy(hx_[i])
                self.memory.append((g_tmp, f_tmp, c_tmp, hx_tmp, a_tmp, reward_temp, g_tmp_, f_tmp_, c_tmp_, hx_tmp_))

                if len(self.memory) > self.memory_size :
                    self.memory.pop(0)
                else:
                    self.memory_size += 1

    def __getitem__(self, index):
        (g, f, c, hx, a, r, g_, f_, c_, hx_) = self.memory[index]
        return g, f, c, hx, a, r, g_, f_, c_, hx_

    def __len__(self):
        return self.memory_size





# if __name__ == '__main__':
    # dataset = Image_preprocess()
    # print(len(dataset))





