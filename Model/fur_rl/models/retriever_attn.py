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

        self.Attn_seq = ModelAttn()
        self.fc_seq0 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc_seq1 = nn.Linear(in_features=512, out_features=1, bias=True)

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


    def forward(self, img, txt, actions, hx_his, r):
        # action [B, 50, 512]
        # img [B, step, 512]
        # txt [B, step, 512]
        x = self.Attn_seq(q=actions, k=txt, v=txt)      # [B, 50, 512]
        p = self.softmax(self.fc_seq0(x))              # [B, 50, 1]
        q = self.logistic(self.fc_seq1(x))              # [B, 50, 1]

        p = torch.clamp(p, min=1e-10, max=1-1e-10)
        return p.squeeze(2), q, hx_his


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
        self.neg_memory = []
        self.memory_counter = 0
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.00001)
        self.loss_func1 = nn.MSELoss()
        self.loss_func2 = nn.MSELoss()
        self.batch_mem = {}
        self.is_store = None

    def store_transition(self, g, f, c, hx, a, r, g_, f_, c_, hx_, t, batch_size, net_mem, success_turn, turn):
        if turn == 0:
            for i in range(batch_size):
                self.batch_mem[i] = []
            self.is_store = torch.zeros((batch_size), device=self.device)
        for i in range(batch_size):
            if r[i] <= -2000:
                continue
            else:
                f_seq = torch.zeros((12, 512), device=self.device)
                f_seq[0:turn+2] = f[:,i]
                f_seq_ = torch.zeros((12, 512), device=self.device)
                f_seq_[0:turn+3] = f_[:,i]
                g_tmp = copy.deepcopy(g[i])
                f_tmp = copy.deepcopy(f_seq)       # [len, 512]
                c_tmp = copy.deepcopy(c[i])
                hx_tmp = copy.deepcopy(hx[i])
                a_tmp = copy.deepcopy(a[i])
                reward_temp = copy.deepcopy(r[i])
                g_tmp_ = copy.deepcopy(g_[i])
                f_tmp_ = copy.deepcopy(f_seq_)
                c_tmp_ = copy.deepcopy(c_[i])
                hx_tmp_ = copy.deepcopy(hx_[i])
                t_tmp = copy.deepcopy(t[i])
                self.batch_mem[i].append((g_tmp, f_tmp, c_tmp, hx_tmp, a_tmp, reward_temp,
                                          g_tmp_, f_tmp_, c_tmp_, hx_tmp_, t_tmp))

                if success_turn[i] > 0 and self.is_store[i] == 0:
                    # print("len", len(self.memory), len(self.batch_mem[i]))
                    self.memory.append(self.batch_mem[i])
                    self.is_store[i] = 1
                    # self.memory.append((g_tmp, f_tmp, c_tmp, hx_tmp, a_tmp, reward_temp,
                    #                       g_tmp_, f_tmp_, c_tmp_, hx_tmp_, t_tmp))
                    # print("len", len(self.memory))
                    while len(self.memory) > net_mem:
                        self.memory.pop(0)
                    self.memory_counter = len(self.memory)
                    # print("success", success_turn[i], reward_temp, len(self.memory))
                elif turn == 9 and self.is_store[i] == 0:
                    self.neg_memory.append(self.batch_mem[i])
                    while len(self.neg_memory) > net_mem+16:
                        self.neg_memory.pop(0)
                    # print("neg", success_turn[i], reward_temp, len(self.neg_memory))

    def learn(self, batch_size, device=None, i=0):
        if len(self.neg_memory) < len(self.memory):
            return 0, 0
        # target parameter update
        actor_loss = 0
        critic_loss = 0
        batch_mem = self.memory[i]
        g, f, c, hx, a, r, g_, f_, c_, hx_, t = zip(*[batch_mem[i] for i in range(len(batch_mem))])
        b_g = torch.stack(g)
        b_f = torch.stack(f)
        b_c = torch.stack(c)
        b_hx = torch.stack(hx)
        b_a = torch.stack(a).unsqueeze(1)
        # b_r = torch.stack(r)
        b_r = torch.zeros((len(r), 1), device=self.device)
        for i in range(len(r)):
            if i == 0:
                b_r[len(r) - i - 1] = r[len(r) - i - 1]
            else:
                b_r[len(r) - i - 1] = 0.5 * b_r[len(r) - i]
        b_g_ = torch.stack(g_)
        b_f_ = torch.stack(f_)
        b_c_ = torch.stack(c_)
        b_hx_ = torch.stack(hx_)
        b_t = torch.stack(t)

        batch_mem = self.neg_memory[i]
        g, f, c, hx, a, r, g_, f_, c_, hx_, t = zip(*[batch_mem[i] for i in range(len(batch_mem))])
        b_g_neg = torch.stack(g)
        b_f_neg = torch.stack(f)
        b_c_neg = torch.stack(c)
        b_hx_neg = torch.stack(hx)
        b_a_neg = torch.stack(a).unsqueeze(1)
        # b_r_neg = torch.stack(r)
        b_r_neg = torch.zeros((len(r), 1), device=self.device)
        for i in range(len(r)):
            if i == 0:
                b_r_neg[len(r) - i - 1] = r[len(r) - i - 1]
            else:
                b_r_neg[len(r) - i - 1] = 0.5 * b_r_neg[len(r) - i]

        b_g_neg_ = torch.stack(g_)
        b_f_neg_ = torch.stack(f_)
        b_c_neg_ = torch.stack(c_)
        b_hx_neg_ = torch.stack(hx_)
        b_t_neg = torch.stack(t)

        b_g = torch.cat((b_g, b_g_neg), 0)
        b_f = torch.cat((b_f, b_f_neg), 0)
        b_c = torch.cat((b_c, b_c_neg), 0)
        b_hx = torch.cat((b_hx, b_hx_neg), 0)
        b_a = torch.cat((b_a, b_a_neg), 0)
        b_r = torch.cat((b_r, b_r_neg), 0)
        b_g_ = torch.cat((b_g_, b_g_neg_), 0)
        b_f_ = torch.cat((b_f_, b_f_neg_), 0)
        b_c_ = torch.cat((b_c_, b_c_neg_), 0)
        b_hx_ = torch.cat((b_hx_, b_hx_neg_), 0)
        b_t = torch.cat((b_t, b_t_neg), 0)


        ## actor loss
        log_probs, q_eval, b_hx_from_net = self.actor_net(b_g, b_f.detach(), b_c, b_hx.detach(), b_r)
        log_probs_temp = torch.zeros((b_a.shape[0], 1)).to(device)
        q_eval_temp = torch.zeros((b_a.shape[0], 1)).to(device)
        for i in range(b_a.shape[0]):
            log_probs_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
            q_eval_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
        q_next = self.actor_net(b_g_, b_f_, b_c_, b_hx_, b_r)[1].detach()
        q_target = b_r + 0.8 * q_next.max(1)[0].view(b_a.shape[0], 1)
        delta = q_target - q_eval_temp
        delta = b_r - q_eval_temp
        # print('q_target', q_target[0], 'q_eval', q_eval_temp[0], 'delta', delta[0])
        print("log_probs_temp", log_probs_temp, "b_r", b_r)
        actor_loss = torch.mean(- torch.log(log_probs_temp) * b_r.detach()).float()
        # critic_loss = torch.mean(delta ** 2).float()
        critic_loss = 0

        ## supervised_loss
        supervised_loss = 0

        ## total loss
        loss = actor_loss + critic_loss + supervised_loss
        self.actor_optimizer.zero_grad()
        print("actor_loss", actor_loss, "critic_loss", critic_loss, "supervised_loss", supervised_loss)
        loss.backward()
        self.actor_optimizer.step()

        # self.memory.pop(0)
        self.neg_memory.pop(0)

        return actor_loss, critic_loss





