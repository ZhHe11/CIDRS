import sys
sys.path.append("/home/data/zh/project1/streamlit")
sys.path.append("/home/data/zh/project1/streamlit/Model")
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from fur_rl.models.retriever_ac import DQN_v3
import matplotlib.pyplot as plt
import copy

# device1 = "cuda:4" if torch.cuda.is_available() else "cpu"
# device2 = "cuda:0" if torch.cuda.is_available() else "cpu
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
device1 = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
resume = '/home/data/zh/project1/SaveModel/ObjEpoch_50.pt'
# this pre-training models are used for user model
obj_clip_model, obj_clip_preprocess = load_from_name("ViT-B-16", device=device1,
                                                     download_root='../../data/pretrained_weights/',
                                                     resume=resume)
# this pre-training models are used for agents
# resume = '/home/data/zh/project1/SaveModel/SentenceEpoch_50.pt'
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
            reward[batch_i] = 1000
            print("batch_i", batch_i, 'succeed')
            continue
        elif turn == max_turn-1:
            reward[batch_i] = -20
            print("this sample failed...")
            continue
        else:
            pass
        rank = ranker.compute_rank(G_next_imgs[batch_i].unsqueeze(0), T_imgs[batch_i].unsqueeze(0))
        # reward[batch_i] = -0.01 * rank
        if rank < 10:
            is_success_10[batch_i] = 1
            reward[batch_i] = 10
        elif rank < 50:
            is_success_100[batch_i] = 1
            reward[batch_i] = -1
        else:
            reward[batch_i] = -10
            pass
    return is_success, is_success_10, is_success_100, reward


def show_img(img_id):
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
            g_fbs.append("")
            t_fbs.append(t_fb_tmp)
        else:
            feedbacks.append("我不要" + g_fb_tmp + "，我想要" + t_fb_tmp)
            g_fbs.append(g_fb_tmp)
            t_fbs.append(t_fb_tmp)
    # print(G_ids, T_ids)
    return g_fbs, t_fbs, feedbacks


def main():
    # sentence_clip_model2 = '005/epoch_50.pt'
    # resume = '/hy-tmp/zh/project/CLIP/Chinese-CLIP-master/Chinese-CLIP-master/usage/save_path' + '/' + sentence_clip_model2
    # resume = '/home/data/zh/project1/SaveModel/SentenceEpoch_50.pt'
    # sentence_clip_model2, sentence_clip_preprocess2 = load_from_name("ViT-B-16", device=device1, download_root='../../data/pretrained_weights/',
    #                                    resume=resume)
    # CLIP.usage.calculate.convert_models_to_fp32(sentence_clip_model2)

    # load retriever model and class ranker
    net = DQN_v3(sentence_clip_model, sentence_clip_preprocess, device=device1)
    # model_path = '../output/dqn-v11.pth'
    # net.eval_net.load_state_dict(torch.load(model_path)['state_dict'])
    # load objs
    with open('./dict_img.json', 'r') as f:
        load_dict_train = json.load(f)
    with open('./dict-test0304.json', 'r') as f:
        load_dict_test = json.load(f)

    # load data
    batch_size = 16
    net_mem = 100
    net_batch = 32
    model_name = "F004/F004-sig-soft"
    turns = 10
    num_actions = 50

    train_img_ids = "./datasets/train_img_id1.csv"
    test_img_ids = "./datasets/test_img_id_r.csv"
    dataset_train = recommadation.datasets.img_preprocess.Image_preprocess(train_img_ids)
    dataset_test = recommadation.datasets.img_preprocess.Image_preprocess(test_img_ids)
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    ranker = utils.ranker_1.Ranker(device1, dataset_train, batch_size=64)
    ranker_test = utils.ranker_1.Ranker(device1, dataset_test, batch_size=64)

    EPISODES = 20

    acc_max = 0
    load_model_name = '/home/data/zh/project1/streamlit/output/' + 'F004/F004-reward-train' + '.pth'
    net.actor_net.load_state_dict(torch.load(load_model_name)['actor_state_dict'])
    net.eval_net.load_state_dict(torch.load(load_model_name)['eval_state_dict'])
    net.actor_optimizer.load_state_dict(torch.load(load_model_name)['actor_optimizer'])
    net.critic_optimizer.load_state_dict(torch.load(load_model_name)['critic_optimizer'])
    update_train, update_test = 0, 0
    for episode in range(EPISODES):
        # test process
        if update_test == 0:
            time1 = time.time()
            ranker_test.update_emb(model=net.actor_net)  # 220.0789999961853s; 78s on 3090
            print("ranker_test update_emb time: ", time.time() - time1)
            update_test = 1
        evaluate(dataloader_test, device1, batch_size, net, ranker_test, load_dict_test, turns, num_actions, model_name,
                 episode)


# def train(dataloader, device1, batch_size, net, ranker, load_dict, turns, num_actions, model_name, episode, acc_max, net_mem, net_batch):
#     net.actor_net.train()
#     net.eval_net.train()
#     EPSOLON = 0.9
#     max_turn = 10
#     # recommendation process
#     # initialize
#     acc_all, acc_all_at10, acc_all_at100 = 0, 0, 0
#     acc_t10, acc_t10_at10, acc_t10_at100 = 0, 0, 0
#     batch_count = 0
#     # begin
#     for data in dataloader:
#         # 0. initialize
#         batch_start_time = time.time()
#         imgs, img_ids = data
#         imgs = imgs.to(device1)
#         T_pre, G_pre = imgs, imgs
#         T_imgs = net.actor_net.img_embed(imgs)
#         if imgs.shape[0] != batch_size:
#             continue
#         T_ids = img_ids.numpy()
#         is_success, is_success_10, is_success_100 = np.zeros(len(T_ids)), np.zeros(len(T_ids)), np.zeros(len(T_ids))
#         success_turn = np.zeros(len(T_ids))
#         reward = np.zeros(len(T_ids))
#         loss_in_turn_a = 0
#         loss_in_turn_c = 0
#         f_embed_his_t = []
#         f_embed_his_g = []
#         f_embed_his = []
#         net.actor_net.init_hid(batch_size)
#         net.actor_net.Stage1.hx = net.actor_net.Stage1.hx.to(device1)
#         hx_in_turn = net.actor_net.Stage1.hx.detach()
#         # 1. turn = -1: init first img and txt
#         g_txt, t_txt, fb_txt = get_feedback_from_json(T_ids, T_ids, turn=-1, load_dict=load_dict,
#                                                       sentence_clip_model=sentence_clip_model, G_pre=G_pre,
#                                                       T_pre=T_pre)
#         with torch.no_grad():
#             f_embed_t = net.actor_net.txt_embed(clip.tokenize(t_txt).to(device1))
#             f_embed_g = net.actor_net.txt_embed(clip.tokenize(g_txt).to(device1))
#             f_embed = net.actor_net.txt_embed(clip.tokenize(fb_txt).to(device1))
#         f_embed_his_t.append(f_embed_t)
#         f_embed_his_g.append(f_embed_g)
#         f_embed_his.append(f_embed)
#         C_ids, C_imgs, C_pre = ranker.candidate(f_embed_his_t, f_embed_his_g, num_actions, -1)  # torch.Size([2, 4]), torch.Size([2, 4, 512])
#         ranker.compute_rank(f_embed_t, T_imgs, "dot")
#         G_ids = C_ids[:, 0].cpu().numpy()
#         G_imgs = C_imgs[:, 0]
#         G_pre = C_pre[:, 0]
#         hx_in_turn = G_imgs
#         g_txt, t_txt, fb_txt = get_feedback_from_json(G_ids, T_ids, turn=0, load_dict=load_dict,
#                                                       sentence_clip_model=sentence_clip_model, G_pre=G_pre,
#                                                       T_pre=T_pre)
#         # print(T_ids, G_ids, fb_txt)
#         with torch.no_grad():
#             f_embed_t = net.actor_net.txt_embed(clip.tokenize(t_txt).to(device1))
#             f_embed_g = net.actor_net.txt_embed(clip.tokenize(g_txt).to(device1))
#             f_embed = net.actor_net.txt_embed(clip.tokenize(fb_txt).to(device1))
#         f_embed_his_t.append(f_embed_t)
#         f_embed_his_g.append(f_embed_g)
#         f_embed_his.append(f_embed)
#         C_ids, C_imgs, C_pre = ranker.candidate(f_embed_his_t, f_embed_his_g, num_actions)
#         is_success, is_success_10, is_success_100, reward = get_reward(
#             batch_size, is_success, is_success_10, is_success_100, G_ids, T_ids, G_imgs, T_imgs, ranker, reward, turn=-1, max_turn=max_turn)
#         acc_in_turn, acc_in_turn_10, acc_in_turn_100, success_turn = get_acc(
#             batch_size, is_success, is_success_10, is_success_100, success_turn, turn=-1)
#
#         acc_t10_in_turn, acc_t10_in_turn_10, acc_t10_in_turn_100 = 0, 0, 0
#         reward = torch.tensor(reward).to(device1).unsqueeze(1).float()
#         if acc_in_turn == batch_size:
#             print("[[[this batch all succeed]]]")
#             continue
#
#         # 2. turn >= 0:
#         for turn in range(max_turn):
#             if acc_in_turn == batch_size:
#                 print("#########[[[this batch all succeed]]]##############")
#                 break
# ##################################################### 3. get action ###################################################
#             out, _ = net.actor_net.forward(G_imgs, f_embed_t, C_imgs, hx_in_turn, reward)
#             print("out1:", out[0])
#             # out = torch.zeros((batch_size, num_actions)).to(device1)
#             # hx_in_turn_tmp = torch.zeros((batch_size, num_actions, 512)).to(device1)
#             # for i in range(num_actions):
#                 # print(G_imgs.shape, f_embed_t.shape, C_imgs[:, i, :].unsqueeze(1).shape, hx_in_turn_e.shape, reward.shape)
#                 # q_temp = net.eval_net(G_imgs, f_embed_t, C_imgs[:, i, :].unsqueeze(1), hx_in_turn_e, reward).detach()
#                 # hx_in_turn_tmp[: i :] = net.eval_net.Stage1.hx.detach().unsqueeze(dim=1)[: i :]
#                 # out[:, i] = q_temp.squeeze(dim=1)
#             hx_next_in_turn = net.actor_net.Stage1.hx.detach()
#             # print("hx_next_in_turn", hx_next_in_turn.shape)acc
#             if random.random() < EPSOLON:
#                 max_index = torch.argmax(out, dim=1)  # tensor([3, 3], device='cuda:3')
#             else:
#                 max_index = torch.zeros(batch_size).long().to(device1)
#                 for batch_i in range(batch_size):
#                     # max_index[batch_i] = random.randint(0, num_actions - 1)    # if greed search, max_index[batch_i] = 0
#                     max_index[batch_i] = 0
#             print("max_index", max_index)
#             G_next_ids = C_ids[torch.arange(len(C_ids)), max_index]  # tensor([549254, 527948], device='cuda:3')
#             G_next_imgs = C_imgs[torch.arange(len(C_imgs)), max_index]
#             G_next_pre = C_pre[torch.arange(len(C_pre)), max_index]
#             # 4. get next state
#             G_next_ids = G_next_ids.cpu().numpy()
#             g_next_txt, t_next_txt, fb_next_txt = get_feedback_from_json(
#                 G_next_ids, T_ids, turn, load_dict=load_dict,
#                 sentence_clip_model=sentence_clip_model, G_pre=G_next_pre, T_pre=T_pre)
#             with torch.no_grad():
#                 f_next_embed_t = net.actor_net.txt_embed(clip.tokenize(t_next_txt).to(device1))
#                 f_next_embed_g = net.actor_net.txt_embed(clip.tokenize(g_next_txt).to(device1))
#                 f_next_embed = net.actor_net.txt_embed(clip.tokenize(fb_next_txt).to(device1))
#             f_embed_his_t.append(f_next_embed_t)
#             f_embed_his_g.append(f_next_embed_g)
#             f_embed_his.append(f_next_embed)
#             C_next_ids, C_next_imgs, C_next_pre = ranker.candidate(f_embed_his_t, f_embed_his_g, num_actions)
#             # 5. get reward
#             is_success, is_success_10, is_success_100, reward = get_reward(
#                 batch_size, is_success, is_success_10, is_success_100, G_ids, T_ids, G_imgs, T_imgs, ranker, reward, turn=turn, max_turn=max_turn)
#             acc_in_turn, acc_in_turn_10, acc_in_turn_100, success_turn = get_acc(
#                 batch_size, is_success, is_success_10, is_success_100, success_turn, turn=turn)
#             if turn == 9:
#                 acc_t10_in_turn, acc_t10_in_turn_10, acc_t10_in_turn_100 = acc_in_turn, acc_in_turn_10, acc_in_turn_100
#             net.store_transition(G_imgs, f_embed_t, C_imgs, hx_in_turn, max_index, reward,
#                                  G_next_imgs, f_next_embed_t, C_next_imgs, hx_next_in_turn, batch_size=batch_size,
#                                  net_mem=net_mem, success_turn=is_success, turn=turn)
#             # 7. train model
#             if len(net.memory) >= net_mem:
#                 actor_loss, critic_loss = net.learn(batch_size=net_batch, device=device1)
#                 loss_in_turn_a += actor_loss.detach().cpu().numpy()
#                 loss_in_turn_c += critic_loss.detach().cpu().numpy()
#             # 8. update state:
#             G_imgs = G_next_imgs
#             G_ids = G_next_ids
#             f_embed_t = f_next_embed_t
#             f_embed_g = f_next_embed_g
#             f_embed = f_next_embed
#             C_imgs = C_next_imgs
#             C_pre = C_next_pre
#             C_ids = C_next_ids
#             hx_in_turn = hx_next_in_turn
#             t_txt = t_next_txt
#             fb_txt = fb_next_txt
#             # print(G_ids, fb_txt)
#             # a = input()
#             # if (turn + 1) % 10 == 0:
#             if acc_in_turn == batch_size:
#                 print("#########[[[this batch all succeed]]]##############")
#                 break
#
#             print("turn: ", turn, "loss", loss_in_turn_a, loss_in_turn_c, "success_turn: ", success_turn, "batch: ", batch_count)
#
#         acc_all += acc_in_turn
#         acc_all_at10 += acc_in_turn_10
#         acc_all_at100 += acc_in_turn_100
#         acc_t10 += acc_t10_in_turn
#         acc_t10_at10 += acc_t10_in_turn_10
#         acc_t10_at100 += acc_t10_in_turn_100
#         batch_count += 1
#         batch_cost_time = time.time() -batch_start_time
#         print("【Batch】", "%d/%d" % (batch_count, len(dataloader)),
#               ' ({:.3f}%)'.format(batch_count / len(dataloader) * 100),
#               "【time_left】", "%.2f" % (batch_cost_time * (len(dataloader) - batch_count) / 60), "min")
#         print("acc: %.4f" % (acc_all / (batch_count * batch_size)),
#               "acc@10: %.4f" % (acc_all_at10 / (batch_count * batch_size)),
#               "acc@50: %.4f" % (acc_all_at100 / (batch_count * batch_size)),
#               "acc_t10: %.4f" % (acc_t10 / (batch_count * batch_size)),
#               "acc_t10@10: %.4f" % (acc_t10_at10 / (batch_count * batch_size)),
#               "acc_t10@100: %.4f" % (acc_t10_at100 / (batch_count * batch_size)),
#               "loss_actor:", loss_in_turn_a / turns, "loss_critic:", loss_in_turn_c / turns)
#         if acc_all > max(0.4, acc_max):
#             acc_max = acc_all
#             state = {
#                 'episode': episode + 1,
#                 'actor_state_dict': net.actor_net.state_dict(),
#                 'eval_state_dict': net.eval_net.state_dict(),
#                 'actor_optimizer': net.actor_optimizer.state_dict(),
#                 'critic_optimizer': net.critic_optimizer.state_dict(),
#             }
#             torch.save(state, "../output/" + model_name + "-train.pth")
#
#     # save result to txt
#     acc_all = acc_all / (batch_count * batch_size)
#     acc_all_at10 = acc_all_at10 / (batch_count * batch_size)
#     acc_all_at100 = acc_all_at100 / (batch_count * batch_size)
#     with open("../output/" + model_name + ".txt", "a") as f:
#         f.write("episode: " + str(episode) + " acc: " + str(acc_all)
#                 + " acc@10: " + str(acc_all_at10) + " acc@50: " + str(acc_all_at100) + "\n")
#     # save model
#     print("[episode: ", str(episode), "] acc: ", str(acc_all),
#           " acc@10: ", str(acc_all_at10), " acc@50: ", str(acc_all_at100))
#     if acc_all > max(0.01, acc_max):
#         acc_max = acc_all
#         state = {
#             'episode': episode + 1,
#             'actor_state_dict': net.actor_net.state_dict(),
#             'eval_state_dict': net.eval_net.state_dict(),
#             'actor_optimizer': net.actor_optimizer.state_dict(),
#             'critic_optimizer': net.critic_optimizer.state_dict(),
#         }
#         torch.save(state, "../output/" + model_name + ".pth")
#     return net, acc_max
#

def evaluate(dataloader, device1, batch_size, net, ranker, load_dict, turns, num_actions, model_name, episode):
    net.actor_net.eval()
    net.eval_net.eval()
    EPSOLON = 1
    max_turn = 10
    # recommendation process
    # initialize
    acc_all, acc_all_at10, acc_all_at100 = 0, 0, 0
    acc_t10, acc_t10_at10, acc_t10_at100 = 0, 0, 0
    batch_count = 0
    # begin
    for data in dataloader:
        # 0. initialize
        batch_start_time = time.time()
        imgs, img_ids = data
        imgs = imgs.to(device1)
        if imgs.shape[0] != batch_size:
            continue
        T_ids = img_ids.numpy()
        # G_ids = T_ids.copy()
        T_pre = imgs
        G_pre = imgs
        T_imgs = net.actor_net.img_embed(imgs)  # [batch_size, 512]
        # G_imgs = torch.zeros(T_imgs.shape).to(device1)  # [batch_size, 512]
        is_success = np.zeros(len(T_ids))
        is_success_10 = np.zeros(len(T_ids))
        is_success_100 = np.zeros(len(T_ids))
        success_turn = np.zeros(len(T_ids))
        reward = np.zeros(len(T_ids))
        loss_in_turn_a = 0
        loss_in_turn_c = 0
        f_embed_his_t = []
        f_embed_his_g = []
        f_embed_his = []
        net.actor_net.init_hid(batch_size)
        net.actor_net.Stage1.hx = net.actor_net.Stage1.hx.to(device1)
        hx_in_turn = net.actor_net.Stage1.hx.detach()
        # print("目标图像：")
        # show_img(T_ids)
        # input()
        # 1. turn = -1:
        ## first feedback fb_(-1)
        g_txt, t_txt, fb_txt = get_feedback_from_json(T_ids, T_ids, turn=-1, load_dict=load_dict,
                                                      sentence_clip_model=sentence_clip_model, G_pre=G_pre,
                                                      T_pre=T_pre)
        # print("第一次输入：")
        # print("g_txt", g_txt)
        # print("t_txt", t_txt)
        # print("fb_txt: ", fb_txt)
        # input()
        with torch.no_grad():
            f_embed_t = net.actor_net.txt_embed(clip.tokenize(t_txt).to(device1))
            f_embed_g = net.actor_net.txt_embed(clip.tokenize(g_txt).to(device1))
            f_embed = net.actor_net.txt_embed(clip.tokenize(fb_txt).to(device1))
        f_embed_his_t.append(f_embed_t)
        f_embed_his_g.append(f_embed_g)
        f_embed_his.append(f_embed)
        ## first Given_img G_0
        C_ids, C_imgs, C_pre = ranker.candidate(f_embed_his_t, f_embed_his_g, num_actions,
                                                -1)  # torch.Size([2, 4]), torch.Size([2, 4, 512])
        ranker.compute_rank(f_embed_t, T_imgs, "dot")
        # print("C_img.shape", C_imgs.shape, C_pre.shape)
        G_ids = C_ids[:, 0].cpu().numpy()
        G_imgs = C_imgs[:, 0]
        G_pre = C_pre[:, 0]
        # print("G_ids", G_ids, G_imgs.shape, G_pre.shape)
        ## generate feedback fb_0 according to G_0 and T
        g_txt, t_txt, fb_txt = get_feedback_from_json(G_ids, T_ids, turn=0, load_dict=load_dict,
                                                      sentence_clip_model=sentence_clip_model, G_pre=G_pre,
                                                      T_pre=T_pre)
        with torch.no_grad():
            f_embed_t = net.actor_net.txt_embed(clip.tokenize(t_txt).to(device1))
            f_embed_g = net.actor_net.txt_embed(clip.tokenize(g_txt).to(device1))
            f_embed = net.actor_net.txt_embed(clip.tokenize(fb_txt).to(device1))
        f_embed_his_t.append(f_embed_t)
        f_embed_his_g.append(f_embed_g)
        f_embed_his.append(f_embed)
        # print("g_txt", g_txt, t_txt, fb_txt)
        C_ids, C_imgs, C_pre = ranker.candidate(f_embed_his_t, f_embed_his_g, num_actions)
        print("[rank]", ranker.compute_rank(f_embed_t, T_imgs, "dot"))
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
##################################################### 3. get action ###################################################
            out, _ = net.actor_net.forward(G_imgs, f_embed_t, C_imgs, hx_in_turn, reward)
            # out = torch.zeros((batch_size, num_actions)).to(device1)
            # for i in range(num_actions):
            #     q_temp = net.eval_net(G_imgs, f_embed_t, C_imgs[:, i, :].unsqueeze(1), hx_in_turn, reward).detach()
            #     out[:, i] = q_temp.squeeze(dim=1)
            print("out:", out[0])
            hx_next_in_turn = net.actor_net.Stage1.hx.detach()
            if random.random() < EPSOLON:
                max_index = torch.argmax(out, dim=1)  # tensor([3, 3], device='cuda:3')
            else:
                max_index = torch.zeros(batch_size).long().to(device1)
                for batch_i in range(batch_size):
                    max_index[batch_i] = random.randint(0, num_actions - 1)  # if greed search,
                    # max_index[batch_i] = 0
            print("[test action]:", max_index)
            G_next_ids = C_ids[torch.arange(len(C_ids)), max_index]  # tensor([549254, 527948], device='cuda:3')
            G_next_imgs = C_imgs[torch.arange(len(C_imgs)), max_index]
            G_next_pre = C_pre[torch.arange(len(C_pre)), max_index]
            # 4. get next state
            G_next_ids = G_next_ids.cpu().numpy()
            g_next_txt, t_next_txt, fb_next_txt = get_feedback_from_json(
                G_next_ids, T_ids, turn, load_dict=load_dict,
                sentence_clip_model=sentence_clip_model, G_pre=G_next_pre, T_pre=T_pre)
            with torch.no_grad():
                f_next_embed_t = net.actor_net.txt_embed(clip.tokenize(t_next_txt).to(device1))
                f_next_embed_g = net.actor_net.txt_embed(clip.tokenize(g_next_txt).to(device1))
                f_next_embed = net.actor_net.txt_embed(clip.tokenize(fb_next_txt).to(device1))
            f_embed_his_t.append(f_next_embed_t)
            f_embed_his_g.append(f_next_embed_g)
            f_embed_his.append(f_next_embed)
            C_next_ids, C_next_imgs, C_next_pre = ranker.candidate(f_embed_his_t, f_embed_his_g, num_actions)
            # 5. get reward
            is_success, is_success_10, is_success_100, reward = get_reward(
                batch_size, is_success, is_success_10, is_success_100, G_ids, T_ids, G_imgs, T_imgs, ranker, reward, turn=turn, max_turn=max_turn)
            acc_in_turn, acc_in_turn_10, acc_in_turn_100, success_turn = get_acc(
                batch_size, is_success, is_success_10, is_success_100, success_turn, turn=turn)
            if turn == 9:
                acc_t10_in_turn, acc_t10_in_turn_10, acc_t10_in_turn_100 = acc_in_turn, acc_in_turn_10, acc_in_turn_100
            if acc_in_turn == batch_size:
                print("#########[[[this batch all succeed]]]##############")
                break
            # 7. train model
            # 8. update state:
            G_imgs = G_next_imgs
            G_ids = G_next_ids
            f_embed_t = f_next_embed_t
            f_embed_g = f_next_embed_g
            f_embed = f_next_embed
            C_imgs = C_next_imgs
            C_pre = C_next_pre
            C_ids = C_next_ids
            hx_in_turn = hx_next_in_turn
            t_txt = t_next_txt
            fb_txt = fb_next_txt
            # print("##learn time:## ", time.time() - start_time)
            if (turn + 1) % 10 == 0:
                print("turn: ", turn, "loss", loss_in_turn_a, loss_in_turn_c, "success_turn: ", success_turn, "batch: ",
                      batch_count)

        acc_all += acc_in_turn
        acc_all_at10 += acc_in_turn_10
        acc_all_at100 += acc_in_turn_100
        acc_t10 += acc_t10_in_turn
        acc_t10_at10 += acc_t10_in_turn_10
        acc_t10_at100 += acc_t10_in_turn_100
        batch_count += 1
        batch_cost_time = time.time() - batch_start_time
        print("[Batch finish] ", "%d/%d" % (batch_count, len(dataloader)),
              ' 【{:.3f}%】'.format(batch_count / len(dataloader) * 100),
              "time_left", batch_cost_time * (len(dataloader) - batch_count) / 60, "min")
        print("episode: ", episode, "batch: ", batch_count,
              "acc: ", acc_all / (batch_count * batch_size),
              "acc@10: ", acc_all_at10 / (batch_count * batch_size),
              "acc@50: ", acc_all_at100 / (batch_count * batch_size),
              "acc_t10: ", acc_t10 / (batch_count * batch_size),
              "acc_t10@10: ", acc_t10_at10 / (batch_count * batch_size),
              "acc_t10@100: ", acc_t10_at100 / (batch_count * batch_size),
              "loss_actor:", loss_in_turn_a / turns, "loss_in_turn_c:", loss_in_turn_c / turns)

    # save result to txt
    acc_all = acc_all / (batch_count * batch_size)
    acc_all_at10 = acc_all_at10 / (batch_count * batch_size)
    acc_all_at100 = acc_all_at100 / (batch_count * batch_size)
    with open("../output/" + model_name + "-test.txt", "a") as f:
        f.write("episode: " + str(episode) + " acc: " + str(acc_all)
                + " acc@10: " + str(acc_all_at10) + " acc@50: " + str(acc_all_at100) + "\n")
    print("[episode: ", str(episode), "] acc: ", str(acc_all),
          " acc@10: ", str(acc_all_at10), " acc@50: ", str(acc_all_at100))
    return

if __name__ == "__main__":
    main()
