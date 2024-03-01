import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.CLIP.cn_clip.clip import load_from_name
import cn_clip.clip as clip
from PIL import Image
from torch.autograd import Variable
from fur_rl.models.StageOne import Stage1
from fur_rl.models.transformer import ModelAttn
import copy
import PIL

# v3 带softmax的模型，actor
class Retriever_v3(nn.Module):
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        super(Retriever_v3, self).__init__()
        self.sentence_clip_model2 = sentence_clip_model2
        self.sentence_clip_preprocess2 = sentence_clip_preprocess2
        self.tokenize = clip.tokenize
        self.Stage1 = Stage1(hid_dim=512)
        self.fc02 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc03 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc04 = nn.Linear(in_features=64, out_features=1, bias=True)
        self.fc12 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc13 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc14 = nn.Linear(in_features=64, out_features=1, bias=True)
        # self.fc1 = nn.Linear(in_features=516, out_features=1, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.logistic = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(num_features=1)
        self.Attn0 = ModelAttn()
        self.Attn1 = ModelAttn()
        self.Attn_his = ModelAttn()
        self.norm = nn.LayerNorm(normalized_shape=512)
        self.norm50 = nn.LayerNorm(normalized_shape=50)

        self.hx_his = None

        self.device = device
    def img_embed(self, img):
        img_embed = self.sentence_clip_model2.encode_image(img)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        return img_embed

    def txt_embed(self, txt_token):
        txt_embed = self.sentence_clip_model2.encode_text(txt_token)
        txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
        return txt_embed

    def get_env_feature(self, img, txt, reward):
        hx = self.Stage1.hx
        x = self.Stage1(img, txt, reward)
        x = x / x.norm(dim=-1, keepdim=True)
        return x, hx

    def init_hid(self, batch_size=1):
        self.hx_his = torch.zeros((batch_size, 1, 512), device=self.device)
        return

    def detach_hid(self):
        self.hx_his = None
        return

    def get_Q(self, img, txt, action, reward, hx_his):
        img = img.unsqueeze(dim=1)
        txt = txt.unsqueeze(dim=1)
        # print("img:\n", img, img.shape)
        # print("txt:\n", txt, txt.shape)
        # env_feature = self.Attn0(img, txt, txt)
        env_feature = txt
        env_feature1 = env_feature / env_feature.norm(dim=-1, keepdim=True)
        # print("hx1 and txt:\n", env_feature1[0] @ txt[0].T)
        hx_his = hx_his.unsqueeze(dim=1)
        # print("hx1 and hx_his\n", env_feature1[0] @ hx_his[0].T)
        hx2 = self.Attn_his(env_feature1, hx_his, hx_his)
        hx2 = hx2 / hx2.norm(dim=-1, keepdim=True)
        # print("hx2 and hx_his:\n", hx2[0] @ hx_his[0].T)
        # print("hx2 and hx1:\n", hx2[0] @ env_feature1[0].T)
        # transformers:
        action = action.unsqueeze(dim=1)
        x = self.Attn1(action, hx2, hx2)    # 16 1 512


        # linear:
        p = torch.zeros((action.shape[0], 1), device=self.device)
        for i in range(action.shape[0]):
            p[i] = hx2[i] @ action[i].T
        # p = self.logistic(self.fc02(x))
        # p = self.logistic(self.fc03(p))
        # p = self.logistic(self.fc04(p))
        q = self.logistic(self.fc12(x))
        # q = self.logistic(self.fc13(q))
        # q = self.logistic(self.fc14(q))
        # p = p.squeeze(dim=1)
        q = q.squeeze(dim=1)
        return (p + 1) / 2, q, hx2


    def forward(self, img, txt, action, hx_his, r):
        P = torch.zeros((action.shape[0], action.shape[1], 1), device=self.device)
        Q = torch.zeros((action.shape[0], action.shape[1], 1), device=self.device)
        b_hx_his_ = torch.zeros((action.shape[0], 1, 512), device=self.device)
        for i in range(action.shape[1]):
            P[:, i], Q[:, i], b_hx_his_ = self.get_Q(img, txt, action[:, i], r, hx_his)
        P = P.squeeze(dim=-1)
        # P = self.softmax(P).squeeze(dim=2)
        P = torch.clamp(P, min=1e-6, max=1-1e-6)
        Q = 60 * Q - 30
        return P, Q, b_hx_his_


MEMORY_CAPACITY = 100
N_STATES = 4
GAMMA = 0.95


class DQN_v3():
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device, MULTI_GPU=False, device_ids=None):
        self.actor_net = Retriever_v3(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.num_train = 0
        self.device = device
        self.MULTI_GPU = MULTI_GPU
        self.device_ids = device_ids

        self.memory = []  # initialize memory
        self.memory_counter = 0
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.0001)
        self.loss_func1 = nn.MSELoss()
        self.loss_func2 = nn.MSELoss()
        self.batch_mem = {}


    def store_transition(self, g, f, c, hx, a, r, g_, f_, c_, hx_, t, batch_size, net_mem, success_turn, turn):
        if turn == 0:
            for i in range(batch_size):
                self.batch_mem[i] = []
        for i in range(batch_size):
            if r[i] <= -2000:
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
                t_tmp = copy.deepcopy(t[i])
                self.batch_mem[i].append((g_tmp, f_tmp, c_tmp, hx_tmp, a_tmp, reward_temp,
                                          g_tmp_, f_tmp_, c_tmp_, hx_tmp_, t_tmp))

                if success_turn[i] > 0:
                    # print("len", len(self.memory), len(self.batch_mem[i]))
                    self.memory.extend(self.batch_mem[i])
                    # self.memory.append((g_tmp, f_tmp, c_tmp, hx_tmp, a_tmp, reward_temp,
                    #                       g_tmp_, f_tmp_, c_tmp_, hx_tmp_, t_tmp))
                    # print("len", len(self.memory))
                    while len(self.memory) > net_mem:
                        self.memory.pop(0)
                    self.memory_counter = len(self.memory)
                    # print("success", success_turn[i], reward_temp, len(self.memory))


    def learn(self, batch_size, device=None):
        # target parameter update
        if self.memory_counter < batch_size:
            return 0, 0
        else:
            sample_index = np.random.choice(len(self.memory), batch_size, replace=False)
        sample_index = torch.LongTensor(sample_index).to(device)
        g, f, c, hx, a, r, g_, f_, c_, hx_, t = zip(*[self.memory[i] for i in sample_index])
        b_g = torch.stack(g)
        b_f = torch.stack(f)
        b_c = torch.stack(c)
        b_hx = torch.stack(hx)
        b_a = torch.stack(a).unsqueeze(1)
        b_r = torch.stack(r)
        b_g_ = torch.stack(g_)
        b_f_ = torch.stack(f_)
        b_c_ = torch.stack(c_)
        b_hx_ = torch.stack(hx_)
        b_t = torch.stack(t)
        ## actor loss
        log_probs, q_eval, b_hx_from_net = self.actor_net(b_g, b_f.detach(), b_c, b_hx.detach(), b_r)
        log_probs_temp = torch.zeros((batch_size, 1)).to(device)
        q_eval_temp = torch.zeros((batch_size, 1)).to(device)
        for i in range(b_a.shape[0]):
            log_probs_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
            q_eval_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
        q_next = self.actor_net(b_g_, b_f_, b_c_, b_hx_, b_r)[1].detach()
        q_target = b_r + 0.8 * q_next.max(1)[0].view(batch_size, 1)
        delta = q_target - q_eval_temp
        delta = b_r - q_eval_temp
        # print('q_target', q_target[0], 'q_eval', q_eval_temp[0], 'delta', delta[0])

        actor_loss = torch.mean(- torch.log(log_probs_temp) * b_r.detach()).float()
        critic_loss = torch.mean(delta ** 2).float()

        ## supervised_loss
        sim_hx2_target = torch.zeros((batch_size, 1)).to(device)
        for i in range(b_hx_from_net.shape[0]):
            sim_hx2_target[i] = b_hx_from_net[i] @ b_t[i].unsqueeze(0).t()
        supervised_loss = nn.CrossEntropyLoss()(sim_hx2_target, torch.ones((batch_size, 1)).to(device))
        loss = actor_loss + critic_loss / 60 + 5 * supervised_loss

        self.actor_optimizer.zero_grad()
        print("actor_loss", actor_loss, "critic_loss", critic_loss, "supervised_loss", supervised_loss)
        loss.backward()
        self.actor_optimizer.step()

        return actor_loss, critic_loss





