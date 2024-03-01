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
        self.fc01 = nn.Linear(in_features=512 * 2, out_features=512, bias=True)
        self.fc02 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.fc03 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc04 = nn.Linear(in_features=64, out_features=1, bias=True)
        self.fc12 = nn.Linear(in_features=512, out_features=256, bias=True)
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
        self.norm = nn.LayerNorm(normalized_shape=512)
        self.norm50 = nn.LayerNorm(normalized_shape=50)
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
        self.Stage1.init_hid(batch_size=batch_size)
        return

    def detach_hid(self):
        self.Stage1.detach_hid()
        return

    def get_Q(self, img, txt, action, reward):
        # env_feature, hx = self.get_env_feature(img, txt, reward)
        img = img.unsqueeze(dim=1)
        txt = txt.unsqueeze(dim=1)
        env_feature = self.Attn0(img, txt, txt)
        env_feature = env_feature / env_feature.norm(dim=-1, keepdim=True)
        # transformers:
        x = self.Attn1(env_feature, action, env_feature)

        # linear:
        p = self.relu(self.fc02(x))
        p = self.relu(self.fc03(p))
        p = self.relu(self.fc04(p))
        p = self.logistic(p).squeeze(dim=1)
        q = self.relu(self.fc12(x))
        q = self.relu(self.fc13(q))
        q = self.relu(self.fc14(q))
        q = self.logistic(q).squeeze(dim=1)
        return p, q

    def forward(self, img, txt, action, hx, r):
        self.Stage1.hx = hx
        P = torch.zeros((action.shape[0], action.shape[1],1), device=self.device)
        Q = torch.zeros((action.shape[0], action.shape[1],1), device=self.device)
        for i in range(action.shape[1]):
            P[:,i], Q[:, i] = self.get_Q(img, txt, action[:,i], r)
        # P = P.squeeze(dim=2)
        P = self.softmax(P).squeeze(dim=2)
        P = torch.clamp(P, min=1e-6, max=1-1e-6)
        Q = 60 * Q - 30
        return P, Q




# v5 critic模型，只用1个candidate
class Retriever_v5(nn.Module):
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        super(Retriever_v5, self).__init__()
        self.sentence_clip_model2 = sentence_clip_model2
        self.sentence_clip_preprocess2 = sentence_clip_preprocess2
        self.tokenize = clip.tokenize
        self.Stage1 = Stage1(hid_dim=512)
        # self.conv1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.conv4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc01 = nn.Linear(in_features=512 * 2, out_features=512, bias=True)
        self.fc02 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc03 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc04 = nn.Linear(in_features=64, out_features=1, bias=True)
        # self.fc1 = nn.Linear(in_features=516, out_features=1, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.logistic = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(num_features=1)
        self.Attn0 = ModelAttn()
        self.Attn1 = ModelAttn()
        self.norm = nn.LayerNorm(normalized_shape=512)
        self.norm50 = nn.LayerNorm(normalized_shape=50)
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
        self.Stage1.init_hid(batch_size=batch_size)
        return

    def detach_hid(self):
        self.Stage1.detach_hid()
        return

    def get_Q(self, img, txt, action, reward):
        # env_feature, hx = self.get_env_feature(img, txt, reward)
        img = img.unsqueeze(dim=1)
        txt = txt.unsqueeze(dim=1)
        env_feature = self.Attn0(img, txt, txt)
        # transformers:
        x = self.Attn1(env_feature, action, env_feature)
        # linear:
        x = self.fc02(x)
        # x = self.logistic(x)
        # x = torch.clamp(x, min=1e-5, max=1-1e-5)
        x = x.squeeze(dim=1)
        return x

    def forward(self, img, txt, action, hx, r):
        self.Stage1.hx = hx
        Q = self.get_Q(img, txt, action, r)
        Q = 60 * self.logistic(Q) - 20
        return Q


class Supervised(nn.Module):
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        super(Supervised, self).__init__()
        self.sentence_clip_model2 = sentence_clip_model2
        self.sentence_clip_preprocess2 = sentence_clip_preprocess2
        self.tokenize = clip.tokenize
        self.Stage1 = Stage1(hid_dim=512)
        # self.conv1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.conv4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc01 = nn.Linear(in_features=512 * 2, out_features=512, bias=True)
        self.fc02 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc03 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc04 = nn.Linear(in_features=64, out_features=1, bias=True)
        # self.fc1 = nn.Linear(in_features=516, out_features=1, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.logistic = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(num_features=1)
        self.Attn0 = ModelAttn()
        self.Attn1 = ModelAttn()
        self.norm = nn.LayerNorm(normalized_shape=512)
        self.norm50 = nn.LayerNorm(normalized_shape=50)
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
        self.Stage1.init_hid(batch_size=batch_size)
        return

    def detach_hid(self):
        self.Stage1.detach_hid()
        return

    def get_Q(self, img, txt, action, reward):
        # env_feature, hx = self.get_env_feature(img, txt, reward)
        img = img.unsqueeze(dim=1)
        txt = txt.unsqueeze(dim=1)
        env_feature = self.Attn0(img, txt, txt)
        print("env_feature:", env_feature.shape, env_feature)
        return env_feature

    def forward(self, img, txt, action, hx, r):
        self.Stage1.hx = hx
        Q = self.get_Q(img, txt, action, r)
        return Q



MEMORY_CAPACITY = 100
N_STATES = 4
GAMMA = 0.95


class DQN_v3():
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device, MULTI_GPU=False, device_ids=None):
        self.eval_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.target_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.actor_net = Retriever_v3(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.supervised_net = Supervised(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.num_train = 0
        self.device = device
        self.MULTI_GPU = MULTI_GPU
        self.device_ids = device_ids

        self.memory = []  # initialize memory
        self.memory_counter = 0
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.supervised_optimizer = torch.optim.Adam(self.supervised_net.parameters(), lr=0.01)
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
                self.batch_mem[i].append((g_tmp, f_tmp, c_tmp, hx_tmp, a_tmp, reward_temp,
                                          g_tmp_, f_tmp_, c_tmp_, hx_tmp_))

                if success_turn[i] >= 0:
                    # print("len", len(self.memory), len(self.batch_mem[i]))
                    self.memory.extend(self.batch_mem[i])
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
        g, f, c, hx, a, r, g_, f_, c_, hx_ = zip(*[self.memory[i] for i in sample_index])
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
        # print("b_g", b_g.shape, b_f.shape, b_c.shape, b_hx.shape, b_a.shape, b_r.shape, b_g_.shape, b_f_.shape, b_c_.shape, b_hx_.shape)
        ## critic loss
        # b_c1 = torch.zeros(b_c.shape[0], 1, b_c.shape[2]).to(device)
        # for i in range(b_a.shape[0]):
        #     b_c1[i][0] = b_c[i][b_a[i][0]].to(device)
        # q_eval = self.actor_net(b_g, b_f.detach(), b_c1, b_hx.detach(), b_r)  # shape (batch, 1)
        # q_next = torch.zeros((batch_size, len(b_c_[0]))).to(device)
        # for i in range(len(b_c_[0])):
        #     q_temp = self.eval_net(b_g_, b_f_, b_c_[:, i, :].unsqueeze(1), b_hx_, b_r).detach()
        #     q_next[:, i] = q_temp.squeeze(dim=1)
        # for j in range(len(b_c_)):
        #     if b_r[j] >= 20:
        #         q_next[j, :] = torch.zeros((1, len(b_c_[0]))).to(device)
        # q_target = b_r + 0.6 * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)
        # print("q_eval, q_target", q_eval[0], q_target[0], b_r[0], q_target[0] - q_eval[0])
        # critic_loss = self.loss_func1(q_eval.float(), q_target.float())
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()
        ## actor loss
        log_probs, q_eval = self.actor_net(b_g, b_f.detach(), b_c, b_hx.detach(), b_r)
        log_probs_temp = torch.zeros((batch_size, 1)).to(device)
        q_eval_temp = torch.zeros((batch_size, 1)).to(device)
        for i in range(b_a.shape[0]):
            log_probs_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
            q_eval_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
        q_next = self.actor_net(b_g_, b_f_, b_c_, b_hx_, b_r)[1].detach()
        q_target = b_r + 0.8 * q_next.max(1)[0].view(batch_size, 1)
        delta = q_target - q_eval_temp
        # print("delta", delta)
        actor_loss = torch.mean(- torch.log(log_probs_temp) * delta.detach()).float()
        critic_loss = torch.mean(delta ** 2).float()
        loss = actor_loss + critic_loss

        self.actor_optimizer.zero_grad()
        print("actor_loss", actor_loss, "critic_loss", critic_loss)
        loss.backward()
        self.actor_optimizer.step()

        return actor_loss, critic_loss





