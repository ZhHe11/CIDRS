import os
import re
import time

import cv2
import sys
import json

import matplotlib.pyplot as plt
import torch
from torch import nn
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


def get_data_from_csv(path):
    data_csv = pd.read_csv(path, encoding='utf-8')
    # print(data_csv)
    pic_id_list = data_csv['pic_id'].values
    seg_id_list = data_csv['seg_id'].values
    object_list = data_csv['object'].values
    segment_list = data_csv['segment'].values
    adj_list = data_csv['adj'].values
    des_list = data_csv['des'].values

    return pic_id_list, seg_id_list, object_list, segment_list, adj_list, des_list

class AttDesDataset(data.Dataset):

    def __init__(self, data_root, dataset_name, img_root, dataset_split='train', transform=None,
                 bert_model='bert-base-chinese',
                 des_len=256, obj_len=8, tgt_len=32
                 ):
        self.images = []
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.transform = transform
        self.img_root = img_root
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.des_len = des_len
        self.obj_len = obj_len
        self.tgt_len = tgt_len
        assert self.transform is not None
        self.pic_id_list, self.seg_id_list, self.object_list, self.segment_list, self.adj_list, self.des_list = \
            get_data_from_csv(self.data_root)
        self.data_csv = pd.read_csv(data_root, encoding='utf-8')
    def get_data_from_csv_by_id(self, id, dict=None):
        pic_id_list = self.data_csv['pic_id'].values
        # {id: des_}
        des_list = self.data_csv['des'].values
        start_time = time.time()
        for i in range(len(pic_id_list)):
            if str(pic_id_list[i]) == str(id):
                # print("find: str(pic_id_list[i]) == str(id)", time.time() - start_time)
                return des_list[i]
        return ""

    def get_img_from_id(self, img_id):
        img_filename = self.img_root
        img_filename = img_filename + '/' + str(img_id) + '.jpg'
        img = Image.open(img_filename)
        if self.transform:
            img = self.transform(img)
        return img

    def encode_text_bert(self, text):
        tokens = []
        tokens.append("[CLS]")
        token_obj = self.tokenizer.tokenize(text)
        for token in token_obj:
            tokens.append(token)
        tokens.append("[SEP]")
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens

    def get_all_from_id(self, img_id, obj_given):
        img_id = str(img_id)
        if img_id[0] == '#':
            des = ""
        else:
            des = self.get_data_from_csv_by_id(img_id)
        img = self.get_img_from_id(img_id)
        des = self.encode_text_bert(des)
        obj_given = self.encode_text_bert(obj_given)
        while(len(des) < self.des_len):
            des.append(100)
        while(len(obj_given) < self.obj_len):
            obj_given.append(0)
        assert len(des) == self.des_len
        return img, torch.from_numpy(np.array(des)), torch.from_numpy(np.array(obj_given))

    def __getitem__(self, idx):
        img_id = self.pic_id_list[idx]
        img = self.get_img_from_id(img_id)

        # des = self.des_list[idx].split('[，,；]')
        des = re.split('，|；', str(self.des_list[idx]))
        masked_des = ""                   # chinese
        for i in range(len(des)):
            if i != int(self.seg_id_list[idx]):
                masked_des = masked_des + des[i] + '  '

        obj = self.object_list[idx]         # chinese
        segment = self.segment_list[idx]    # chinese
        masked_des = self.encode_text_bert(masked_des)
        obj = self.encode_text_bert(obj)
        segment = self.encode_text_bert(segment)
        while(len(masked_des) < self.des_len):
            masked_des.append(100)
        while(len(obj) < self.obj_len):
            obj.append(0)
        while(len(segment) < self.tgt_len):
            segment.append(0)

        assert len(masked_des) == self.des_len
        assert len(obj) == self.obj_len
        assert len(segment) == self.tgt_len
        return img, np.array(masked_des), np.array(obj), np.array(segment), img_id

    def __len__(self):
        return len(self.pic_id_list)



if __name__ == '__main__':
    data_root = r'E:\data\Download\fur\dataset\data_for_test1.csv'
    split_root = ''
    dataset_name = 'Furniture'
    #
    # get_data_from_csv(data_root)
    # img_id = 550709
    # img = get_img_from_id(img_id)
    # plt.imshow(img)
    # plt.show()
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    dataset = AttDesDataset(data_root, dataset_name, transform=transforms.Compose([
                                            transforms.Resize((448,448)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
    img, masked_des, obj, segment = dataset.__getitem__(100)

    img_show = np.zeros((len(img[0]), len(img[0][0]), 3))
    img_show[:, :, 0] = img[0]
    img_show[:, :, 1] = img[1]
    img_show[:, :, 2] = img[2]
    plt.imshow(img_show)
    plt.show()
    print(masked_des, len(masked_des))
    print(obj, len(obj))
    print(segment, len(segment))
    print(dataset.__len__())

    # sentence_for_test = "原木地板的厚实与白色纱幔的轻飘营造朴素和浪漫的氛围，而一张编织餐椅灵动轻巧"
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # tokenizer.tokenize(sentence_for_test)
    # print(tokenizer.tokenize(sentence_for_test))
    # print(tokenizer.convert_tokens_to_ids(sentence_for_test))

