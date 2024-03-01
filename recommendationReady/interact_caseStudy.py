import sys
sys.path.append("/home/data/zh/project1/streamlit")
sys.path.append("/home/data/zh/project1/streamlit/Model")
import os
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import CLIP.usage.calculate
import AttDes.validate_local
from fur_rl.models.retriever import Retriever
from Model.CLIP.cn_clip.clip import load_from_name
import cn_clip.clip as clip
import random
import time
import utils.ranker_1
import pandas as pd
import recommadation.datasets.img_preprocess
import json
# from fur_rl.models.retriever import DQN, DQN_v2, DQN_v3
from fur_rl.models.retriever_rl import DQN_v3
import matplotlib.pyplot as plt
import copy

# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
device1 = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
resume = '/home/data/zh/project1/SaveModel/ObjEpoch_50.pt'
# this pre-training models are used for user model
obj_clip_model, obj_clip_preprocess = load_from_name("ViT-B-16", device=device1,
                                                     download_root='../../data/pretrained_weights/',
                                                     resume=resume)
# this pre-training models are used for agents
resume = '/home/data/zh/project1/CLIP/Chinese-CLIP-master/Chinese-CLIP-master/usageCLIP/save_path/005/epoch_16.pt'
sentence_clip_model, sentence_clip_preprocess = load_from_name("ViT-B-16", device=device1,
                                                               download_root='../../data/pretrained_weights/',
                                                               resume=resume)
CLIP.usage.calculate.convert_models_to_fp32(obj_clip_model)
sentence_clip_model = CLIP.usage.calculate.convert_models_to_fp32(sentence_clip_model)

def get_acc(batch_size, is_success, is_success_10, is_success_100, success_turn, turn):
    acc_in_turn = 0
    acc_in_turn_10 = 0
    acc_in_turn_100 = 0
    for i in range(batch_size):
        if is_success[i] == 1:
            acc_in_turn += 1
            if success_turn[i] == 0:
                success_turn[i] = turn + 1
        if is_success_10[i] == 1:
            acc_in_turn_10 += 1
        if is_success_100[i] == 1:
            acc_in_turn_100 += 1
    return acc_in_turn, acc_in_turn_10, acc_in_turn_100, success_turn


def get_reward(batch_size, is_success, is_success_10, is_success_100, G_next_ids, T_ids, G_next_imgs, T_imgs, ranker, reward, turn=0, max_turn=0):
    for batch_i in range(batch_size):
        if is_success[batch_i] == 1:
            reward[batch_i] = -10000  # 成功之后，不再进入记录
            continue

        elif G_next_ids[batch_i] == T_ids[batch_i]:
            is_success[batch_i] = 1
            reward[batch_i] = 10
            # print("batch_i", batch_i, 'succeed')
            continue
        elif turn == max_turn-1:
            reward[batch_i] = -2
            # print("this sample failed...")
            continue
        else:
            pass
        rank = ranker.compute_rank(G_next_imgs[batch_i].unsqueeze(0), T_imgs[batch_i].unsqueeze(0))
        # reward[batch_i] = (-rank + 500) / 500
        if rank < 10:
            is_success_10[batch_i] = 1
            reward[batch_i] = 2
        elif rank < 50:
            is_success_100[batch_i] = 1
            reward[batch_i] = -1
        else:
            reward[batch_i] = -2
            pass
    return is_success, is_success_10, is_success_100, reward


def show_img(img_id):
    print(img_id)
    img_path = "/home/data/zh/fur/processed_img/" + str(img_id)[1:-1] + ".jpg"
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.show()


def get_feedback(G_pre, T_pre, G_ids, T_ids, objs_token, objs_txt, obj_clip_model, AttDes_model, dataset_all,tokenizer, device, sentence_clip_model, turn):
    objs_index = CLIP.usage.calculate.get_obj(obj_clip_model, G_pre, T_pre, objs_token, 5, 10)
    objs = objs_txt[objs_index]
    out_list = []

    label_txt, output1, output2, output3 = AttDes.validate_local.validate_one_img(AttDes_model, dataset_all, G_ids.tolist(), objs, device, tokenizer)
    out_list.append([label_txt, output1, output2, output3])
    label_txt, output1, output2, output3 = AttDes.validate_local.validate_one_img(AttDes_model, dataset_all, T_ids.tolist(), objs, device, tokenizer)
    out_list.append([label_txt, output1, output2, output3])
    g_txt = [out_list[0][1], out_list[0][2], out_list[0][3]]    # ['铺地板则以黑白地毯毛方案吊灯线', '落窗帘皮质单人椅与之形成风格拼接友'], ['背地板上铺贴的地板', '将绿色窗帘带入清爽'], ['墙与地板形成层次晰分明区域划分区域', '书窗帘也选用墨绿色幔来增加空间的清新感']
    t_txt = [out_list[1][1], out_list[1][2], out_list[1][3]]
    probs_all, text_all, feed_backs, G_fbs, T_fbs = CLIP.usage.calculate.calculate_similarity(
        model=sentence_clip_model, img1=G_pre, img2=T_pre, texts1=g_txt, texts2=t_txt, device=device, turn=turn)

    return T_fbs


def get_feedback_from_json(G_ids, T_ids, turn, load_dict, sentence_clip_model, G_pre, T_pre):
    start_time = time.time()
    feedbacks = []
    g_fbs = []
    t_fbs = []
    batch_size = len(G_ids)
    g_fb = []
    t_fb = []
    for i in range(len(G_ids)):
        G_fbs = load_dict[str(G_ids[i])]
        T_fbs = load_dict[str(T_ids[i])]
        g_objs = list(G_fbs.keys())
        t_objs = list(T_fbs.keys())
        for j in range(len(list(G_fbs.keys()))):
            g_fb.append(G_fbs[g_objs[j]].split(",")[0])
            t_fb.append(T_fbs[t_objs[j]].split(",")[0])

    g_fb_token = clip.tokenize(g_fb)     # batch_size*10 * 64
    t_fb_token = clip.tokenize(t_fb)
    img_pre = torch.cat((G_pre, T_pre), dim=0).to(device1)
    txt_token = torch.cat((g_fb_token, t_fb_token), dim=0).to(device1)
    sentence_clip_model.eval()
    sentence_clip_model = sentence_clip_model.to(device1)
    with torch.no_grad():
        image_features, text_features, logit_scale = sentence_clip_model(img_pre, txt_token)
    logits_per_text = logit_scale * text_features @ image_features.t()
    max_index = 0
    for i in range(batch_size):
        b_sim_g_g = logits_per_text[10*i:10*i+10,i]
        b_sim_g_t = logits_per_text[10*i:10*i+10,i + batch_size]
        max_index = torch.argmax(b_sim_g_g - b_sim_g_t)
        g_fb_tmp = g_fb[10*i + max_index]
        b_sim_t_g = logits_per_text[10*(i+batch_size):10*(i+batch_size)+10,i]
        b_sim_t_t = logits_per_text[10*(i+batch_size):10*(i+batch_size)+10,i + batch_size]
        max_index = torch.argmax(b_sim_t_t - b_sim_t_g)
        t_fb_tmp = t_fb[10*i + max_index]
        if turn == -1:
            feedbacks.append("我想要" + t_fb_tmp)
            g_fbs.append(" ")
            t_fbs.append(t_fb_tmp)
        else:
            feedbacks.append("我不要" + g_fb_tmp + "，我想要" + t_fb_tmp)
            if max(b_sim_g_g - b_sim_g_t) > max(b_sim_t_t - b_sim_t_g):
                t_fb_tmp = " "
            elif max(b_sim_g_g - b_sim_g_t) < max(b_sim_t_t - b_sim_t_g):
                g_fb_tmp = " "
            g_fbs.append(g_fb_tmp)
            t_fbs.append(t_fb_tmp)

    return g_fbs, t_fbs, feedbacks


def get_feedback_from_json_real(G_ids, T_ids, turn, load_dict, sentence_clip_model, G_pre, T_pre):
    start_time = time.time()
    feedbacks = []
    g_fbs = []
    t_fbs = []
    batch_size = len(G_ids)
    for i in range(len(G_ids)):
        g_fb = []
        t_fb = []
        G_fbs = load_dict[str(G_ids[i])]
        T_fbs = load_dict[str(T_ids[i])]
        g_objs = list(G_fbs.keys())
        t_objs = list(T_fbs.keys())
        for j in range(len(list(G_fbs.keys()))):
            g_fb.append(G_fbs[g_objs[j]].split(",")[0])
        for j in range(len(list(T_fbs.keys()))):
            t_fb.append(T_fbs[t_objs[j]].split(",")[0])

        g_fb_token = clip.tokenize(g_fb)  # batch_size*10 * 64
        t_fb_token = clip.tokenize(t_fb)
        img_pre = torch.cat((G_pre[i].unsqueeze(0), T_pre[i].unsqueeze(0)), dim=0).to(device1)
        txt_token = torch.cat((g_fb_token, t_fb_token), dim=0).to(device1)
        with torch.no_grad():
            image_features, text_features, logit_scale = sentence_clip_model(img_pre, txt_token)
            logits_per_text = logit_scale * text_features @ image_features.t()
        delta_fb_g = logits_per_text[0:g_fb_token.shape[0], 0] - logits_per_text[0:g_fb_token.shape[0], 1]
        max_index = torch.argmax(delta_fb_g)
        g_fb_tmp = g_fb[max_index]
        delta_fb_t = logits_per_text[g_fb_token.shape[0]:, 1] - logits_per_text[g_fb_token.shape[0]:, 0]
        max_index = torch.argmax(delta_fb_t)
        t_fb_tmp = t_fb[max_index]
        if turn == -1:
            max_index = torch.argmax(logits_per_text[g_fb_token.shape[0]:, 1] - 0)
            t_fb_tmp = t_fb[max_index]
            feedbacks.append("我想要" + t_fb_tmp)
            g_fbs.append(" ")
            t_fbs.append(t_fb_tmp)
        else:
            feedbacks.append("我不要" + g_fb_tmp + "，我想要" + t_fb_tmp)
            if max(delta_fb_g) > max(delta_fb_t):
                t_fb_tmp = " "
            elif max(delta_fb_g) < max(delta_fb_t):
                g_fb_tmp = " "
            g_fbs.append(g_fb_tmp)
            t_fbs.append(t_fb_tmp)
    return g_fbs, t_fbs, feedbacks



def txt_embed(t_txt, g_txt, fb_txt, net, batch_size, device1):
    with torch.no_grad():
        f_embed_t = net.actor_net.txt_embed(clip.tokenize(t_txt).to(device1))
        f_embed_g = net.actor_net.txt_embed(clip.tokenize(g_txt).to(device1))
        for i in range(batch_size):
            if len(t_txt[i]) == 0:
                f_embed_t[i] = torch.zeros((1, 512)).to(device1)
            if len(g_txt[i]) == 0:
                f_embed_g[i] = torch.zeros((1, 512)).to(device1)
        f_embed = net.actor_net.txt_embed(clip.tokenize(fb_txt).to(device1))
    return f_embed_t, f_embed_g, f_embed




def main():
    net = DQN_v3(sentence_clip_model, sentence_clip_preprocess, device=device1)
    # load objs
    with open('./datajson/dict_img.json', 'r') as f:
        load_dict_train = json.load(f)
    with open('./datajson/dict-test0304.json', 'r') as f:
        load_dict_test = json.load(f)
    with open('./datajson/dict-real.json', 'r') as f:
        load_dict_test_r = json.load(f)

    # load data
    batch_size = 1
    net_batch = 2
    model_name = "F017/F017-2p-t10-11"
    # model_name = "F016/F016-1p-r"
    turns = 10
    num_actions = 4

    train_img_ids = "./datasets/train_img_id1.csv"
    test_img_ids = "./datasets/test_img_id_r.csv"
    # all_img_ids = "./datasets/imgs_large.csv"
    all_img_ids = "./datasets/train_large.csv"
    dataset_train = recommadation.datasets.img_preprocess.Image_preprocess(train_img_ids)
    dataset_test = recommadation.datasets.img_preprocess.Image_preprocess(test_img_ids)
    dataset_all = recommadation.datasets.img_preprocess.Image_preprocess(all_img_ids)
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_all = torch.utils.data.DataLoader(dataset_all, batch_size=batch_size, shuffle=False, num_workers=4)
    ranker = utils.ranker_1.Ranker(device1, dataset_train, batch_size=64)
    ranker_test = utils.ranker_1.Ranker(device1, dataset_test, batch_size=64)
    ranker_all = utils.ranker_1.Ranker(device1, dataset_all, batch_size=64)

    EPISODES = 20

    acc_max = 0
    load_model_name = '/home/data/zh/project1/streamlit/output/' + 'F015/F015-1p-t10-g' + '.pth'
    # load_model_name = '/home/data/zh/project1/streamlit/output/' + 'F017/F017-2p-t10-17-g' + '.pth'
    net.actor_net.load_state_dict(torch.load(load_model_name)['actor_state_dict'])
    net.actor_optimizer.load_state_dict(torch.load(load_model_name)['actor_optimizer'])
    update_train, update_test = 0, 0
    acc_test_max_g = 0
    acc_test_max_r = 0
    for episode in range(EPISODES):
        # # # train process
        if update_train == 0:
            time1 = time.time()
            ranker.update_emb(model=net.actor_net)  # 220.0789999961853s; 78s on 3090
            print("ranker_all update_emb time: ", time.time() - time1)
            # save ranker_all
            # torch.save(ranker_all, './ranker_all.pth')
            # print("ranker_all saved")
            # load ranker_all
            # ranker_all = torch.load('./ranker_all.pth', map_location=device1)
            update_train = 1
        net, acc_max = train(dataloader_all, device1, batch_size, net, ranker, load_dict_train, turns, num_actions, model_name,
                             episode, acc_max, net_batch)
        # test process
        if update_test == 0:
            time1 = time.time()
            ranker_test.update_emb(model=net.actor_net)  # 220.0789999961853s; 78s on 3090
            print("ranker_test update_emb time: ", time.time() - time1)
            update_test = 1
        net, acc_test_max_g = train(dataloader_test, device1, batch_size, net, ranker_test, load_dict_test, turns, num_actions, model_name,
              episode, acc_max, net_batch, isTest=True, isReal=False, acc_test_max=acc_test_max_g)
        # net, acc_test_max_r = train(dataloader_test, device1, batch_size, net, ranker_test, load_dict_test_r, turns, num_actions, model_name,
        #       episode, acc_max, net_batch, isTest=True, isReal=True, acc_test_max=acc_test_max_r)


def train(dataloader, device1, batch_size, net, ranker, load_dict, turns, num_actions, model_name, episode, acc_max, net_batch, isTest=False, isReal=False, acc_test_max=0):
    lensOfFeedbacks = 12
    EPSOLON = 1
    net.actor_net.eval()
    max_turn = turns
    # recommendation process
    # initialize
    acc_all, acc_all_at10, acc_all_at100 = 0, 0, 0
    acc_t10, acc_t10_at10, acc_t10_at100 = 0, 0, 0
    success_turn_all = []
    batch_count = 0
    # begin
    for data in dataloader:
        # 0. initialize
        batch_start_time = time.time()
        imgs, img_ids = data
        imgs = imgs.to(device1)
        T_pre, G_pre = imgs, imgs
        T_imgs = net.actor_net.img_embed(imgs).detach()
        if imgs.shape[0] != batch_size:
            continue
        T_ids = img_ids.numpy()
        T_index = torch.zeros((batch_size, 1)).to(device1)
        for i in range(batch_size):
            for j in range(len(ranker.data_id)):
                if ranker.data_id[j] == T_ids[i]:
                    T_index[i] = j
                    break
        is_success, is_success_10, is_success_100 = np.zeros(len(T_ids)), np.zeros(len(T_ids)), np.zeros(len(T_ids))
        success_turn = np.zeros(len(T_ids))
        reward = np.zeros(len(T_ids))
        loss_in_turn_a = 0
        loss_in_turn_c = 0
        reward_list = []
        print("g_txt, t_txt, reward")
        g_txt = input()
        t_txt = input()
        fb_txt = input()
        if g_txt == "":
            g_txt = " "
        if t_txt == "":
            t_txt = " "
        if fb_txt == "":
            fb_txt = '0'
        reward_list.append(int(fb_txt))
        f_embed_t, f_embed_g, f_embed = txt_embed(t_txt, g_txt, fb_txt, net, batch_size, device1)
        f_embed_his_t = torch.zeros((batch_size, lensOfFeedbacks, 512)).to(device1)
        f_embed_his_g = torch.zeros((batch_size, lensOfFeedbacks, 512)).to(device1)
        f_embed_his = torch.zeros((batch_size, lensOfFeedbacks, 512)).to(device1)
        f_embed_his_t[:, 0] = f_embed_t
        f_embed_his_g[:, 0] = f_embed_g
        f_embed_his[:, 0] = f_embed

        C_ids, C_imgs, C_pre = ranker.candidate_fb(f_embed_his, f_embed_his_t, f_embed_his_g, num_actions, -1, batch_size)  # torch.Size([2, 4]), torch.Size([2, 4, 512])
        G_ids = C_ids[:, 0].cpu().numpy()
        G_imgs = C_imgs[:, 0]
        G_pre = C_pre[:, 0]
        G_imgs_his = torch.zeros((batch_size, lensOfFeedbacks, 512)).to(device1)
        G_imgs_his[:, 0] = G_imgs
        show_img(G_ids)
        print("g_txt, t_txt, fb_txt")
        g_txt = input()
        t_txt = input()
        fb_txt = input()
        if g_txt == "":
            g_txt = " "
        if t_txt == "":
            t_txt = " "
        if fb_txt == "":
            fb_txt = '0'
        reward_list.append(int(fb_txt))
        f_embed_t, f_embed_g, f_embed = txt_embed(t_txt, g_txt, fb_txt, net, batch_size, device1)
        f_embed_his_t[:, 1] = f_embed_t
        f_embed_his_g[:, 1] = f_embed_g
        f_embed_his[:, 1] = f_embed
        # C_ids, C_imgs, C_pre = ranker.candidate_fb(f_embed_his, f_embed_his_t, f_embed_his_g, num_actions)
        distance_, distance_t, distance_g = ranker.actions_metric(f_embed_his, f_embed_his_t, f_embed_his_g, batch_size, lensOfFeedbacks)
        is_success, is_success_10, is_success_100, reward = get_reward(
            batch_size, is_success, is_success_10, is_success_100, G_ids, T_ids, G_imgs, T_imgs, ranker, reward, turn=-1, max_turn=max_turn)
        acc_in_turn, acc_in_turn_10, acc_in_turn_100, success_turn = get_acc(
            batch_size, is_success, is_success_10, is_success_100, success_turn, turn=-1)

        acc_t10_in_turn, acc_t10_in_turn_10, acc_t10_in_turn_100 = 0, 0, 0
        reward = torch.tensor(reward).to(device1).unsqueeze(1).float()
        if acc_in_turn == batch_size:
            print("[[[this batch all succeed]]]")
            continue

        # 2. turn >= 0:
        for turn in range(max_turn):
            if acc_in_turn == batch_size:
                print("#########[[[this batch all succeed]]]##############")
                break
##################################################### 3. get action ###################################################
            # img, txt, actions_matric, ranker
            p, q, maxC_img, maxC_ids, C_imgs, C_ids, C_pre, p_db = net.actor_net.forward(
                G_imgs_his, torch.cat((f_embed_his_t, f_embed_his_g), dim=1), torch.cat((distance_t, -distance_g), dim=1),
                ranker, k=num_actions)
            if turn == turns - 1:
                max_index = torch.zeros(batch_size).long().to(device1)
                for batch_i in range(batch_size):
                    max_index[batch_i] = 0
                print("final action: ", max_index)
            elif random.random() < EPSOLON:
                # max_index = p.argmax(dim=1)
                max_index = torch.zeros(batch_size).long().to(device1)
                for batch_i in range(batch_size):
                    max_index[batch_i] = 0
                print("net action: ", max_index)
            else:
                pass

            G_next_ids = torch.zeros(batch_size).to(device1)
            G_next_imgs = torch.zeros(batch_size, 512).to(device1)
            G_next_pre = torch.zeros(batch_size, 3, 224, 224).to(device1)

            for i in range(batch_size):
                # random action
                # index = random.randint(0, ranker.data_id.shape[0] - 1)
                # G_next_ids[i] = ranker.data_id[index]
                # G_next_imgs[i] = ranker.data_emb[index]
                # G_next_pre[i] = ranker.data_pre[index]
                G_next_ids[i] = C_ids[i][max_index[i]]        # [16]
                G_next_imgs[i] = C_imgs[i][max_index[i]]      # [16, 512]
                G_next_pre[i] = C_pre[i][max_index[i]]       # [16, 3, 224, 224]
            G_next_ids = G_next_ids.cpu().numpy().astype(int)
            show_img(G_next_ids)
            G_imgs_his_next = G_imgs_his
            G_imgs_his_next[:, turn + 1] = G_next_imgs
            G_next_index = torch.zeros((batch_size, 1)).to(device1)
            print("g_txt, t_txt, fb_txt")
            g_next_txt = input()
            t_next_txt = input()
            fb_next_txt = input()
            if g_next_txt == "":
                g_next_txt = " "
            if t_next_txt == "":
                t_next_txt = " "
            if fb_next_txt == "":
                fb_next_txt = '0'
            reward_list.append(int(fb_next_txt))
            f_next_embed_t, f_next_embed_g, f_next_embed = txt_embed(t_next_txt, g_next_txt, fb_next_txt, net, batch_size, device1)
            f_embed_his_t_next, f_embed_his_g_next, f_embed_his_next = f_embed_his_t, f_embed_his_g, f_embed_his
            f_embed_his_t_next[:, turn+2] = f_next_embed_t
            f_embed_his_g_next[:, turn+2] = f_next_embed_g
            f_embed_his_next[:, turn+2] = f_next_embed
            distance_next_, distance_next_t, distance_next_g = ranker.actions_metric(f_embed_his_next, f_embed_his_t_next, f_embed_his_g_next,
                                                                                    batch_size, lensOfFeedbacks)
            # 5. get reward
            is_success, is_success_10, is_success_100, reward = get_reward(
                batch_size, is_success, is_success_10, is_success_100, G_next_ids, T_ids, G_next_imgs, T_imgs, ranker, reward, turn=turn, max_turn=max_turn)
            acc_in_turn, acc_in_turn_10, acc_in_turn_100, success_turn = get_acc(
                batch_size, is_success, is_success_10, is_success_100, success_turn, turn=turn)

            # 8. update state:
            G_imgs = G_next_imgs
            G_ids = G_next_ids
            G_imgs_his = G_imgs_his_next
            f_embed = f_next_embed
            t_txt = t_next_txt
            fb_txt = fb_next_txt
            f_embed_his_t = f_embed_his_t_next
            f_embed_his_g = f_embed_his_g_next
            f_embed_his = f_embed_his_next
            distance_t = distance_next_t
            distance_g = distance_next_g
            distance_ = distance_next_
            if acc_in_turn == batch_size:
                print("#########[[[this batch all succeed]]]##############")
                break
            print("turn: ", turn, "loss", loss_in_turn_a, loss_in_turn_c, "success_turn: ", success_turn, "batch: ", batch_count)
        file_reward = open("reward.txt", "a")
        for i in range(len(reward_list)):
            # save reward
            file_reward.write(str(reward_list[i]) + " ")
        file_reward.write("\n")

        acc_all += acc_in_turn
        acc_all_at10 += acc_in_turn_10
        acc_all_at100 += acc_in_turn_100
        acc_t10 += acc_t10_in_turn
        acc_t10_at10 += acc_t10_in_turn_10
        acc_t10_at100 += acc_t10_in_turn_100
        success_turn_tmp = [x for x in success_turn if x != 0]
        success_turn_all += success_turn_tmp
        AST = -1
        if len(success_turn_all) > 0:
            AST = success_turn_all
        batch_count += 1
        batch_cost_time = time.time() - batch_start_time
        print("【Batch】", "%d/%d" % (batch_count, len(dataloader)),
              ' ({:.3f}%)'.format(batch_count / len(dataloader) * 100),
              "【time_left】", "%.2f" % (batch_cost_time * (len(dataloader) - batch_count) / 60), "min")
        print("acc: %.4f" % (acc_all / (batch_count * batch_size)),
              "AverageSuccessTurn: %.4f" % (np.array(AST).mean()),
              "acc@10: %.4f" % (acc_all_at10 / (batch_count * batch_size)),
              "acc@50: %.4f" % (acc_all_at100 / (batch_count * batch_size)),
              "loss_actor:", loss_in_turn_a / turns, "loss_critic:", loss_in_turn_c / turns)

    # save result to txt
    acc_all = acc_all / (batch_count * batch_size)
    acc_all_at10 = acc_all_at10 / (batch_count * batch_size)
    acc_all_at100 = acc_all_at100 / (batch_count * batch_size)
    success_turn_all = np.array(success_turn_all).mean()
    with open("../output/" + model_name + ".txt", "a") as f:
        if isTest == True:
            f.write('[' + model_name + ']' + str(episode) + " acc: " + str(acc_all) + " aSTurn: " + str(success_turn_all)
                    + " acc@10: " + str(acc_all_at10) + " acc@50: " + str(acc_all_at100) + "\n")
        else:
            f.write("train: " + str(episode) + " acc: " + str(acc_all) + " aSTurn: " + str(success_turn_all)
                    + " acc@10: " + str(acc_all_at10) + " acc@50: " + str(acc_all_at100) + "\n")
    # save model
    print("[episode: ", str(episode), "] acc: ", str(acc_all),
          " acc@10: ", str(acc_all_at10), " acc@50: ", str(acc_all_at100), " acc_max: ", str(acc_max))
    state = {
        'episode': episode + 1,
        'actor_state_dict': net.actor_net.state_dict(),
        'actor_optimizer': net.actor_optimizer.state_dict(),
    }
    if isTest and acc_all > acc_test_max:
        acc_test_max = acc_all
        if isReal:
            torch.save(state, "../output/" + model_name + "-r.pth")
        else:
            torch.save(state, "../output/" + model_name + "-g.pth")
        return net, acc_test_max
    if isTest == False:
        return net, acc_max
    return net, acc_test_max

def evaluate_epoch(load_name):
    net = DQN_v3(sentence_clip_model, sentence_clip_preprocess, device=device1)
    # model_path = '../output/dqn-v11.pth'
    # net.eval_net.load_state_dict(torch.load(model_path)['state_dict'])
    # load objs
    with open('./dict-test0304.json', 'r') as f:
        load_dict_test = json.load(f)

    # load data
    batch_size = 20
    model_name = "test"
    turns = 10
    num_actions = 8

    train_img_ids = "./datasets/train_img_id1.csv"
    test_img_ids = "./datasets/test_img_id_r.csv"
    dataset_train = recommadation.datasets.img_preprocess.Image_preprocess(train_img_ids)
    dataset_test = recommadation.datasets.img_preprocess.Image_preprocess(test_img_ids)
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    ranker = utils.ranker_1.Ranker(device1, dataset_train, batch_size=64)
    ranker_test = utils.ranker_1.Ranker(device1, dataset_test, batch_size=64)

    load_model_name = '/home/data/zh/project1/streamlit/output/' + load_name + '.pth'
    net.actor_net.load_state_dict(torch.load(load_model_name)['actor_state_dict'])
    net.actor_optimizer.load_state_dict(torch.load(load_model_name)['actor_optimizer'])
    update_train, update_test = 0, 0
    for episode in range(1):
        if update_test == 0:
            time1 = time.time()
            ranker_test.update_emb(model=net.actor_net)  # 220.0789999961853s; 78s on 3090
            print("ranker_test update_emb time: ", time.time() - time1)
            update_test = 1
        train(dataloader_test, device1, batch_size, net, ranker_test, load_dict_test, turns, num_actions, model_name,
              episode, 0, 0, 0, isTest=True)


if __name__ == "__main__":
    main()
