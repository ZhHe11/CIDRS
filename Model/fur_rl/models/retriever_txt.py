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
        self.sentence_clip_model_grads = sentence_clip_model2
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
        self.bn = nn.BatchNorm1d(num_features=1)
        self.Attn0 = ModelAttn()
        self.Attn1 = ModelAttn()
        self.Attn_his = ModelAttn()
        self.norm = nn.LayerNorm(normalized_shape=512)
        self.norm50 = nn.LayerNorm(normalized_shape=50)

        self.Attn_seq = ModelAttn()
        self.fc_seq0 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc_seq1 = nn.Linear(in_features=512, out_features=1, bias=True)

        self.Attn_cross = ModelAttn()
        self.fc_q = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc_p = nn.Linear(in_features=512, out_features=1, bias=True)

        self.hx_his = None
        self.tanh = nn.Tanh()
        self.logistic = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.device = device
    def img_embed(self, img):
        img_embed = self.sentence_clip_model2.encode_image(img)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        return img_embed

    def txt_embed(self, txt_token):
        txt_embed = self.sentence_clip_model2.encode_text(txt_token)
        txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
        return txt_embed

    def txt_embed_grads(self, txt_token):
        txt_embed = self.sentence_clip_model_grads.encode_text(txt_token)
        txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
        return txt_embed

    def weight_train(self, img, txt_tokens, txt_matric, actions_matric, ranker, k=50):
        # to do: use img and txt to do the fusion
        p_db, topk_score, topk_action, candidates, candidates_id, candidates_pre, weight_txt = \
            self.get_candidates(txt_matric, actions_matric, k=k, ranker=ranker)
        return weight_txt, p_db


    def get_candidates_greedy(self, txt, actions_matric, k, ranker):
        x = self.Attn_seq(q=txt, k=txt, v=txt)      # [batch, 24, 512]
        x = self.tanh(self.fc_seq0(x))         # [batch, 24, 1]
        x = x.squeeze(dim=2).unsqueeze(dim=1)       # [batch, 1, 24]
        # for baseline
        x = torch.ones(x.shape, device=self.device)     # [batch, 1, 24]
        # for baseline
        weight_txt = x
        x = torch.bmm(x, actions_matric).squeeze(dim=1)  # actions_matric[batch, 24, 2000] x [batch, 2000]
        # normalize
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
        p = self.logistic(x)
        p = torch.clamp(p, min=1e-10, max=1 - 1e-10)        # [batch, 2000]
        topk_score, topk_action = torch.topk(p, k=k, dim=-1)      # [batch, k] [batch, k]
        candidates = torch.zeros((txt.shape[0], k, 512), device=self.device)    # [batch, k, 512]
        candidates_pre = torch.zeros((txt.shape[0], k, 3, 224, 224), device=self.device)    # [batch, k, 3, 224, 224]
        candidates_id = torch.zeros((txt.shape[0], k), device=self.device)    # [batch, k]
        for i in range(k):
            candidates[:, i, :] = ranker.data_emb[topk_action[:, i], :]    # [batch, 1, 512]
            candidates_pre[:, i, :, :, :] = ranker.data_pre[topk_action[:, i], :, :, :]    # [batch, 1, 3, 224, 224]
            candidates_id[:, i] = ranker.data_id[topk_action[:, i]]    # [batch, 1]

        return p, topk_score, topk_action, candidates, candidates_id, candidates_pre, weight_txt

    def get_candidates(self, txt, actions_matric, k, ranker):
        x = self.Attn_seq(q=txt, k=txt, v=txt)      # [batch, 24, 512]
        x = self.tanh(self.fc_seq0(x))         # [batch, 24, 1]
        x = x.squeeze(dim=2).unsqueeze(dim=1)       # [batch, 1, 24]
        # for baseline
        # x = torch.ones(x.shape, device=self.device)     # [batch, 1, 24]
        # for baseline
        weight_txt = x
        x = torch.bmm(x, actions_matric).squeeze(dim=1)  # actions_matric[batch, 24, 2000] x [batch, 2000]
        # normalize
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
        p = self.logistic(x)
        p = torch.clamp(p, min=1e-10, max=1 - 1e-10)        # [batch, 2000]
        topk_score, topk_action = torch.topk(p, k=k, dim=-1)      # [batch, k] [batch, k]
        candidates = torch.zeros((txt.shape[0], k, 512), device=self.device)    # [batch, k, 512]
        candidates_pre = torch.zeros((txt.shape[0], k, 3, 224, 224), device=self.device)    # [batch, k, 3, 224, 224]
        candidates_id = torch.zeros((txt.shape[0], k), device=self.device)    # [batch, k]
        for i in range(k):
            candidates[:, i, :] = ranker.data_emb[topk_action[:, i], :]    # [batch, 1, 512]
            candidates_pre[:, i, :, :, :] = ranker.data_pre[topk_action[:, i], :, :, :]    # [batch, 1, 3, 224, 224]
            candidates_id[:, i] = ranker.data_id[topk_action[:, i]]    # [batch, 1]

        return p, topk_score, topk_action, candidates, candidates_id, candidates_pre, weight_txt


    def action_greedy(self, img, txt_tokens, txt_matric, actions_matric, ranker, k=50):
        # to do: use img and txt to do the fusion
        # txt [batch, turn+2, 52]
        txt_eb = torch.zeros((txt_tokens.shape[0], txt_tokens.shape[1], 512), device=self.device)
        for i in range(txt_tokens.shape[1]):
            txt_eb[:, i, :] = self.txt_embed_grads(txt_tokens[:, i, :].long())
        # txt [batch, 12, 512]
        fusion = torch.cat((txt_eb, img), dim=1)     # [batch, 25, 512]
        p_db, topk_score, topk_action, candidates, candidates_id, candidates_pre, weight_txt = \
            self.get_candidates_greedy(txt_matric, actions_matric, k=k, ranker=ranker)
        x = self.Attn_cross(q=candidates, k=fusion, v=fusion)      # [batch, k, 512]
        # to do: use MLP replace linear
        p = self.softmax(self.fc_p(x)).squeeze(dim=2)       # [batch, k]
        p = torch.clamp(p, min=1e-10, max=1 - 1e-10)
        # to do: use MLP replace linear
        q = self.tanh(self.fc_q(x)).squeeze(dim=2)       # [batch, k]

        p_maxIndex = p.argmax(dim=-1)        # [batch]
        max_candidates = candidates[torch.arange(candidates.shape[0]), p_maxIndex]   # [batch, 512]
        max_candidates_id = candidates_id[torch.arange(candidates_id.shape[0]), p_maxIndex]   # [batch]

        return p, q, max_candidates, max_candidates_id, candidates, candidates_id, candidates_pre, p_db


    def forward(self, img, txt_tokens, txt_matric, actions_matric, ranker, k=50):
        # to do: use img and txt to do the fusion
        # txt [batch, turn+2, 52]
        txt_eb = torch.zeros((txt_tokens.shape[0], txt_tokens.shape[1], 512), device=self.device)
        for i in range(txt_tokens.shape[1]):
            txt_eb[:, i, :] = self.txt_embed_grads(txt_tokens[:, i, :].long())
        # txt [batch, 12, 512]
        fusion = torch.cat((txt_eb, img), dim=1)     # [batch, 25, 512]
        p_db, topk_score, topk_action, candidates, candidates_id, candidates_pre, weight_txt = \
            self.get_candidates(txt_matric, actions_matric, k=k, ranker=ranker)
        x = self.Attn_cross(q=candidates, k=fusion, v=fusion)      # [batch, k, 512]
        # to do: use MLP replace linear
        p = self.softmax(self.fc_p(x)).squeeze(dim=2)       # [batch, k]
        p = torch.clamp(p, min=1e-10, max=1 - 1e-10)
        # to do: use MLP replace linear
        q = self.tanh(self.fc_q(x)).squeeze(dim=2)       # [batch, k]

        p_maxIndex = p.argmax(dim=-1)        # [batch]
        max_candidates = candidates[torch.arange(candidates.shape[0]), p_maxIndex]   # [batch, 512]
        max_candidates_id = candidates_id[torch.arange(candidates_id.shape[0]), p_maxIndex]   # [batch]

        return p, q, max_candidates, max_candidates_id, candidates, candidates_id, candidates_pre, p_db





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

    def store_transition(self, g, f, d, a, r, g_, f_, d_, t, txt, txt_, batch_size, net_mem, success_turn, turn):
        if turn == 0:
            for i in range(batch_size):
                self.batch_mem[i] = []
            self.is_store = torch.zeros((batch_size), device=self.device)
        for i in range(batch_size):
            if r[i] <= -2000:
                continue
            else:
                g_tmp = copy.deepcopy(g[i])
                f_tmp = copy.deepcopy(f[i])       # [len, 512]
                d_tmp = copy.deepcopy(d[i])
                a_tmp = copy.deepcopy(a[i])
                reward_temp = copy.deepcopy(r[i])
                g_tmp_ = copy.deepcopy(g_[i])
                f_tmp_ = copy.deepcopy(f_[i])
                d_tmp_ = copy.deepcopy(d_[i])
                t_tmp = copy.deepcopy(t[i])
                txt_tmp = copy.deepcopy(txt[i])
                txt_tmp_ = copy.deepcopy(txt_[i])
                self.batch_mem[i].append((g_tmp, f_tmp, d_tmp, a_tmp, reward_temp,
                                          g_tmp_, f_tmp_, d_tmp_, t_tmp, txt_tmp, txt_tmp_))

                if success_turn[i] > 0 and self.is_store[i] == 0:
                    # print("len", len(self.memory), len(self.batch_mem[i]))
                    self.memory.append(self.batch_mem[i])
                    self.is_store[i] = 1
                    while len(self.memory) > net_mem:
                        self.memory.pop(0)
                    self.memory_counter = len(self.memory)
                elif turn == 9 and self.is_store[i] == 0:
                    self.neg_memory.append(self.batch_mem[i])
                    while len(self.neg_memory) > net_mem:
                        self.neg_memory.pop(0)
                    # print("neg", success_turn[i], reward_temp, len(self.neg_memory))

    def learn(self, batch_size, device=None, ranker=None, k=50, batch_count=0):
        success_turn = 0
        fail_turn = 0
        if len(self.memory) == 0 and len(self.neg_memory) == 0:
            return 0,0
        if len(self.memory) > 0:
            success_turn = 1
            batch_mem = self.memory[0]
            g, f, d, a, r, g_, f_, d_, t, txt, txt_ = zip(*[batch_mem[i] for i in range(len(batch_mem))])
            b_g = torch.stack(g)
            b_f = torch.stack(f)
            b_d = torch.stack(d)
            b_a = torch.stack(a).unsqueeze(1)
            b_reward = torch.stack(r)
            b_r = torch.zeros((len(r), 1), device=self.device)
            for i in range(len(r)):
                if i == 0:
                    b_r[len(r) - i - 1] = r[len(r) - i - 1]
                else:
                    b_r[len(r) - i - 1] = r[len(r) - i - 1] + 0.5 * b_r[len(r) - i]
            b_g_ = torch.stack(g_)
            b_f_ = torch.stack(f_)
            b_d_ = torch.stack(d_)
            b_t = torch.stack(t)
            b_txt = torch.stack(txt)
            b_txt_ = torch.stack(txt_)
            self.memory.pop(0)

        if len(self.neg_memory) > 0:
            fail_turn = 1
            batch_mem = self.neg_memory[0]
            g, f, d, a, r, g_, f_, d_, t, txt, txt_ = zip(*[batch_mem[i] for i in range(len(batch_mem))])
            b_g_neg = torch.stack(g)
            b_f_neg = torch.stack(f)
            b_d_neg = torch.stack(d)
            b_a_neg = torch.stack(a).unsqueeze(1)
            b_reward_neg = torch.stack(r)
            b_r_neg = torch.zeros((len(r), 1), device=self.device)
            for i in range(len(r)):
                if i == 0:
                    b_r_neg[len(r) - i - 1] = r[len(r) - i - 1]
                else:
                    b_r_neg[len(r) - i - 1] = r[len(r) - i - 1] + 0.1 * b_r_neg[len(r) - i]
            b_g_neg_ = torch.stack(g_)
            b_f_neg_ = torch.stack(f_)
            b_d_neg_ = torch.stack(d_)
            b_t_neg = torch.stack(t)
            b_txt_neg = torch.stack(txt)
            b_txt_neg_ = torch.stack(txt_)
            self.neg_memory.pop(0)
        if success_turn > 0 and fail_turn > 0:
            b_g = torch.cat((b_g, b_g_neg), 0)
            b_f = torch.cat((b_f, b_f_neg), 0)
            b_d = torch.cat((b_d, b_d_neg), 0)
            b_a = torch.cat((b_a, b_a_neg), 0)
            b_r = torch.cat((b_r, b_r_neg), 0)
            b_g_ = torch.cat((b_g_, b_g_neg_), 0)
            b_f_ = torch.cat((b_f_, b_f_neg_), 0)
            b_d_ = torch.cat((b_d_, b_d_neg_), 0)
            b_t = torch.cat((b_t, b_t_neg), 0)
            b_reward = torch.cat((b_reward, b_reward_neg), 0)
            b_txt = torch.cat((b_txt, b_txt_neg), 0)
        elif success_turn == 0 and fail_turn > 0:
            b_g = b_g_neg
            b_f = b_f_neg
            b_d = b_d_neg
            b_a = b_a_neg
            b_r = b_r_neg
            b_g_ = b_g_neg_
            b_f_ = b_f_neg_
            b_d_ = b_d_neg_
            b_t = b_t_neg
            b_reward = b_reward_neg
            b_txt = b_txt_neg
        elif success_turn > 0 and fail_turn == 0:
            b_g = b_g
            b_f = b_f
            b_d = b_d
            b_a = b_a
            b_r = b_r
            b_g_ = b_g_
            b_f_ = b_f_
            b_d_ = b_d_
            b_t = b_t
            b_reward = b_reward
            b_txt = b_txt
        ## actor loss
        # img, txt, actions_matric, ranker, k = 50
        p, q, max_candidates, max_candidates_id, candidates, candidates_id, candidates_pre, p_db = \
            self.actor_net(b_g, b_txt, b_f, b_d, ranker, k)
        log_probs = torch.zeros(p.shape[0], 1, device=self.device)
        q_eval = torch.zeros(q.shape[0], 1, device=self.device)
        for i in range(p.shape[0]):
            log_probs[i][0] = p[i][b_a[i][0]].to(device)
            q_eval[i][0] = q[i][b_a[i][0]].to(device)
        # q_eval_temp = torch.zeros((b_a.shape[0], 1)).to(device)
        # for i in range(b_a.shape[0]):
        #     log_probs_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
        #     q_eval_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
        # q_next = self.actor_net(b_g_, b_f_, b_c_, b_hx_, b_r)[1].detach()
        # q_target = b_r + 0.8 * q_next.max(1)[0].view(b_a.shape[0], 1)
        # delta = q_target - q_eval_temp
        # delta = b_r - q_eval_temp
        # print('q_target', q_target[0], 'q_eval', q_eval_temp[0], 'delta', delta[0])
        # print("log_probs_temp", max_score, "b_r", b_r, "log(p)", torch.log(log_probs_temp))
        # actor_loss = - (torch.log(log_probs).t() @ b_r.detach()).float()
        actor_loss = 0
        # critic_loss = torch.mean(delta ** 2).float()
        # critic_loss = torch.mean((b_r/5 - q_eval) ** 2).float()
        critic_loss = 0

        ## supervised_loss
        if success_turn == 1:
            weight, p_db_w = self.actor_net.weight_train(b_g, b_txt, b_f, b_d, ranker, k)
            supervised_loss = torch.zeros(1, device=self.device)
            for i in range(b_reward.shape[0]):
                print(b_reward[i][0], p_db_w[i][b_t[i][0]])
                s_loss_tmp = - b_reward[i][0] * p_db_w[i][b_t[i][0]]
                supervised_loss += s_loss_tmp
            supervised_loss = supervised_loss.float()
        else:
            supervised_loss = 0
        ## total loss
        loss = 0.1 * actor_loss + 10 * critic_loss + 10 * supervised_loss
        if loss != 0:
            self.actor_optimizer.zero_grad()
            print("actor_loss", actor_loss, "critic_loss", critic_loss, "supervised_loss", supervised_loss)
            loss.backward()
            self.actor_optimizer.step()

        return actor_loss, critic_loss





