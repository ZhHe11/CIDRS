import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import time

import Model.CLIP.cn_clip.clip as clip
from Model.CLIP.cn_clip.clip import load_from_name, available_models
import random

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.to(torch.float32)
        if p.grad:
            p.grad.data = p.grad.data.to(torch.float32)
    return model


def calculate_similarity(model, img1, img2, texts1, texts2, device, turn=0):
    batch_size = len(img1)
    probs_all = []
    text_all = []
    feed_backs = []
    G_fbs = []
    T_fbs = []
    for i in range(batch_size):
        img1_i = img1[i].unsqueeze(0).to(device)
        img2_i = img2[i].unsqueeze(0).to(device)
        image = torch.cat((img1_i, img2_i), dim=0).to(device)
        text = []
        for j in range(len(texts1)):
            text.append(texts1[j][i])
        for j in range(len(texts2)):
            text.append(texts2[j][i])
        text_all.append(text)
        text_token = clip.tokenize(text).to(device)
        # model
        model.eval()
        image_features, text_features, logit_scale = model(image, text_token)
        # print(image_features, text_features, logit_scale)
        logit_scale = logit_scale.mean()
        # print("logit_scale:", logit_scale)
        logits_per_text = logit_scale * text_features @ image_features.t()

        probs = logits_per_text.detach().softmax(dim=-1).cpu().numpy()
        # print("Label probs:\n", np.around(probs,3))  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
        probs = np.around(probs,3)
        probs_all.append(probs)

        if turn == 0:
            G_fb = ""
            T_fb = text[3]
            feed_back = "我想要" + T_fb + "。"
            feed_backs.append(feed_back)
            G_fbs.append(G_fb)
            T_fbs.append(T_fb)

        else:
            G_fb = ""
            delta = -1
            for i in [2,1,0]:
                if probs[i][0] - probs[i][1] > 0:
                    if text[i] == "":
                        continue
                    elif delta < probs[i][0] - probs[i][1]:
                        delta = probs[i][0] - probs[i][1]
                        G_fb = text[i]
            T_fb = ""
            delta = -1
            for i in [5,4,3]:
                if text[i] == "":
                    continue
                elif delta <= probs[i][1] - probs[i][0]:
                    delta = probs[i][1] - probs[i][0]
                    T_fb = text[i]

            if len(G_fb) != 0 and len(T_fb) != 0:
                feed_back = "我不要" + G_fb + "，我想要" + T_fb + "。"
            elif len(G_fb) != 0 and len(T_fb) == 0:
                feed_back = "我不要" + G_fb + "。"
            elif len(G_fb) == 0 and len(T_fb) != 0:
                feed_back = "我想要" + T_fb + "。"
            else:
                feed_back = "换一个。"
            feed_backs.append(feed_back)
            G_fbs.append(G_fb)
            T_fbs.append(T_fb)
    return probs_all, text_all, feed_backs, G_fbs, T_fbs

def calculate_similarity_one(model, img1, texts1, device, objs, dict_id, T_ids):
    batch_size = len(img1)
    for i in range(batch_size):
        dict_text = dict_id[T_ids[i].cpu().item()]
        img1_i = img1[i].unsqueeze(0).to(device)
        text = []
        text3 = ""
        for j in range(len(texts1)):
            text.append(texts1[j][i])
            text3 += texts1[j][i] + ','
        text_token = clip.tokenize(text).to(device)
        # print("text:", text)
        # model
        model.eval()
        image_features, text_features, logit_scale = model(img1_i, text_token)
        logit_scale = logit_scale.mean()
        logits_per_text = logit_scale * text_features @ image_features.t()
        text_score = logits_per_text.detach().cpu().numpy()
        # print(objs)
        dict_text[objs[i]] = text3
        # print(dict_text)
        # for j in range(len(text)):
        #     print(text[j], text_score[j][0])

    return dict_id

def get_obj(model, G, T, objs, top_k1, top_k2):
    # model
    model.eval()
    image_features, text_features, logit_scale = model(G, objs)
    logits_per_text = logit_scale * text_features @ image_features.t()
    top_list = logits_per_text.detach().cpu().topk(top_k1, dim=0)[1]
    obj_pick = np.random.choice(range(top_k1))
    obj_index1 = top_list[obj_pick]
    image_features, text_features, logit_scale = model(T, objs)
    logits_per_text = logit_scale * text_features @ image_features.t()
    top_list = logits_per_text.detach().cpu().topk(top_k2, dim=0)[1]
    obj_pick = np.random.choice(range(top_k2))
    obj_index2 = top_list[obj_pick]

    random_pick = np.random.choice(range(4))
    if random_pick == 0:
        obj_index = obj_index2
    else:
        obj_index = obj_index1
    return obj_index

def get_objs(model, T, objs, top_k2):
    # model
    model.eval()

    image_features, text_features, logit_scale = model(T, objs)
    logits_per_text = logit_scale * text_features @ image_features.t()
    top_list = logits_per_text.detach().cpu().topk(top_k2, dim=0)

    return top_list[0], top_list[1]


if __name__ == "__main__":

    # texts1 = ["白色吊顶通过金属和黑线勾边打造出几何层叠的效果",
    #             "灰地毯的中性色调与床品、窗帘以深浅对比",
    #             "白灰色床品搭配灰色地毯"]
    texts1 = ["空间","客厅","卧室","墙面","餐厅","公寓","住宅","沙发","家具","地毯","厨房","书房","背景墙","吊灯","墙",
           "卫生间","儿童","床品","装饰","壁纸","地板","窗帘","吊顶","餐椅","别墅","地面","结构","布艺","餐桌","画"]
    texts2 = [
        "打造了一个现代、讲究的温馨的空间。",
        "与地毯的图案",
        "灰地毯的中性色调与床品、窗帘以深浅对比",
        "白灰色床品搭配灰色地毯"
    ]

    print(get_clip_score('902/epoch_50.pt', 423943, texts1))
    # print(texts1, probs1)
    # for i in range(len(texts1)):
    #     print(probs1[i], texts1[i])
    # for i in range(len(texts2)):
    #     print(probs2[i], texts2[i])

