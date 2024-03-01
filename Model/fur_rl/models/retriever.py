import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.CLIP.cn_clip.clip import load_from_name
import cn_clip.clip as clip
from PIL import Image
from torch.autograd import Variable
from fur_rl.models.StageOne import Stage1
import copy
import PIL

class Retriever(nn.Module):
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        super(Retriever, self).__init__()
        self.sentence_clip_model2 = sentence_clip_model2
        self.sentence_clip_preprocess2 = sentence_clip_preprocess2
        self.tokenize = clip.tokenize
        self.Stage1 = Stage1(hid_dim=512)
        self.conv1 = nn.Conv1d(in_channels=52, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc1 = nn.Linear(in_features=516, out_features=50, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.device = device
    def img_embed(self, img):
        img_embed = self.sentence_clip_model2.encode_image(img)
        return img_embed

    def txt_embed(self, txt_token):
        txt_embed = self.sentence_clip_model2.encode_text(txt_token)
        return txt_embed

    def get_env_feature(self, img, txt):
        hx = self.Stage1.hx
        x = self.Stage1(img, txt)
        return x, hx

    def init_hid(self, batch_size=1):
        self.Stage1.init_hid(batch_size=batch_size)
        return

    def detach_hid(self):
        self.Stage1.detach_hid()
        return

    def get_Q(self, img, txt, action):
        env_feature, hx = self.get_env_feature(img, txt)
        env_feature = env_feature.unsqueeze(dim=1)
        hx = hx.unsqueeze(dim=1)
        # cat env_feature and action
        x = torch.cat((env_feature, action, hx), dim=1)
        # convolution layer
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # fully connected layer
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # softmax
        # x = self.softmax(x)
        return x

    def forward(self, img, txt, action, hx):
        self.Stage1.hx = hx
        Q = self.get_Q(img, txt, action)
        return Q

# v2 是训练encode的大模型，训练效率太低
class Retriever_v2(nn.Module):
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        super(Retriever_v2, self).__init__()
        self.sentence_clip_model2 = sentence_clip_model2
        self.sentence_clip_preprocess2 = sentence_clip_preprocess2
        self.tokenize = clip.tokenize
        self.Stage1 = Stage1(hid_dim=512)
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc1 = nn.Linear(in_features=516, out_features=20, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.device = device
    def img_embed(self, img):
        img_embed = self.sentence_clip_model2.encode_image(img)
        return img_embed

    def txt_embed(self, txt_token):
        txt_embed = self.sentence_clip_model2.encode_text(txt_token)
        return txt_embed

    def get_env_feature(self, img, txt):
        hx = self.Stage1.hx
        x = self.Stage1(img, txt)
        return x, hx

    def init_hid(self, batch_size=1):
        self.Stage1.init_hid(batch_size=batch_size)
        return

    def detach_hid(self):
        self.Stage1.detach_hid()
        return

    def get_Q(self, img, txt, action):
        env_feature, hx = self.get_env_feature(img, txt)
        env_feature = env_feature.unsqueeze(dim=1)
        hx = hx.unsqueeze(dim=1)
        # cat env_feature and action
        x = torch.cat((env_feature, action, hx), dim=1)
        # convolution layer
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # fully connected layer
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # softmax
        # x = self.softmax(x)
        return x

    def forward(self, img, txt, action, hx):
        # img txt 是最原始的图片和文本
        self.Stage1.hx = hx
        img_embed = self.img_embed(img)     # 2 512
        txt_embed = self.txt_embed(self.tokenize(txt).to(self.device))
        # print("txt_embed", img_embed.shape, txt_embed.shape)
        c = []
        for i in range(action.shape[1]):
            action_i = self.img_embed(action[:, i])
            c.append(action_i)
        c = torch.stack(c, dim=1)   # 2 20 512
        Q = self.get_Q(img_embed, txt_embed, c)
        return Q
# v3 带softmax的模型，actor
class Retriever_v3(nn.Module):
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        super(Retriever_v3, self).__init__()
        self.sentence_clip_model2 = sentence_clip_model2
        self.sentence_clip_preprocess2 = sentence_clip_preprocess2
        self.tokenize = clip.tokenize
        self.Stage1 = Stage1(hid_dim=512)
        self.conv1 = nn.Conv1d(in_channels=52, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(in_features=516, out_features=50, bias=True)
        self.bn5 = nn.BatchNorm1d(50)
        # self.fc2 = nn.Linear(in_features=516, out_features=50, bias=True)
        # self.fc3 = nn.Linear(in_features=516, out_features=10, bias=True)

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

    def get_env_feature(self, img, txt, reward):
        hx = self.Stage1.hx
        x = self.Stage1(img, txt, reward)
        return x, hx

    def init_hid(self, batch_size=1):
        self.Stage1.init_hid(batch_size=batch_size)
        return

    def detach_hid(self):
        self.Stage1.detach_hid()
        return

    def get_Q(self, img, txt, action, reward):
        env_feature, hx = self.get_env_feature(img, txt, reward)
        env_feature = env_feature.unsqueeze(dim=1)
        hx = hx.unsqueeze(dim=1)
        # cat env_feature and action
        x = torch.cat((env_feature, action, hx), dim=1)
        # convolution layer
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        # fully connected layer
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn5(x)
        # softmax
        x = self.softmax(x)
        return x

    def forward(self, img, txt, action, hx, r):
        self.Stage1.hx = hx
        Q = self.get_Q(img, txt, action, r)
        return Q

# v4 删除了hx输入后面卷积，测试hx对记忆的影响
class Retriever_v4(nn.Module):
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        super(Retriever_v4, self).__init__()
        self.sentence_clip_model2 = sentence_clip_model2
        self.sentence_clip_preprocess2 = sentence_clip_preprocess2
        self.tokenize = clip.tokenize
        self.Stage1 = Stage1(hid_dim=512)
        self.conv1 = nn.Conv1d(in_channels=51, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc1 = nn.Linear(in_features=516, out_features=50, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.device = device
    def img_embed(self, img):
        img_embed = self.sentence_clip_model2.encode_image(img)
        return img_embed

    def txt_embed(self, txt_token):
        txt_embed = self.sentence_clip_model2.encode_text(txt_token)
        return txt_embed

    def get_env_feature(self, img, txt):
        hx = self.Stage1.hx
        x = self.Stage1(img, txt)
        return x, hx

    def init_hid(self, batch_size=1):
        self.Stage1.init_hid(batch_size=batch_size)
        return

    def detach_hid(self):
        self.Stage1.detach_hid()
        return

    def get_Q(self, img, txt, action):
        env_feature, hx = self.get_env_feature(img, txt)
        env_feature = env_feature.unsqueeze(dim=1)
        hx = hx.unsqueeze(dim=1)
        # cat env_feature and action
        x = torch.cat((env_feature, action), dim=1)
        # convolution layer
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # fully connected layer
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # softmax
        x = self.softmax(x)
        return x

    def forward(self, img, txt, action, hx):
        self.Stage1.hx = hx
        Q = self.get_Q(img, txt, action)
        return Q
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
        self.fc02 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.fc03 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc04 = nn.Linear(in_features=64, out_features=1, bias=True)
        # self.fc1 = nn.Linear(in_features=516, out_features=1, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=512, bias=True)
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
        env_feature, hx = self.get_env_feature(img, txt, reward)
        env_feature = env_feature.unsqueeze(dim=1)
        hx = hx.unsqueeze(dim=1)
        # cat env_feature and action
        action = action.squeeze(dim=1)
        action = self.fc2(action)
        action = action.unsqueeze(dim=1)
        # x = torch.cat((env_feature, action, hx), dim=1)
        # print(env_feature.shape, action.shape)
        x = torch.cat((env_feature, action), dim=2)
        # convolution layer
        # print(x.shape)
        x = self.fc01(x)
        # x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.fc02(x)
        x = self.fc03(x)
        x = self.fc04(x)
        x = x.squeeze(dim=1)
        # print(x.shape)
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        # x = self.relu(self.conv4(x))
        # fully connected layer
        # x = x.view(x.shape[0], -1)
        # x = self.fc1(x)
        return x

    def forward(self, img, txt, action, hx, r):
        self.Stage1.hx = hx
        Q = self.get_Q(img, txt, action, r)
        return Q
# v6 训练一个选择candidate的模型
class Retriever_v6(nn.Module):
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        super(Retriever_v6, self).__init__()
        self.sentence_clip_model2 = sentence_clip_model2
        self.sentence_clip_preprocess2 = sentence_clip_preprocess2
        self.tokenize = clip.tokenize
        self.Stage1 = Stage1(hid_dim=512)
        self.conv1 = nn.Conv1d(in_channels=52, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc1 = nn.Linear(in_features=516, out_features=50, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.device = device
    def img_embed(self, img):
        img_embed = self.sentence_clip_model2.encode_image(img)
        return img_embed

    def txt_embed(self, txt_token):
        txt_embed = self.sentence_clip_model2.encode_text(txt_token)
        return txt_embed

    def get_env_feature(self, img, txt):
        hx = self.Stage1.hx
        x = self.Stage1(img, txt)
        return x, hx

    def init_hid(self, batch_size=1):
        self.Stage1.init_hid(batch_size=batch_size)
        return

    def detach_hid(self):
        self.Stage1.detach_hid()
        return

    def get_Q(self, img, txt, action):
        env_feature, hx = self.get_env_feature(img, txt)
        env_feature = env_feature.unsqueeze(dim=1)
        hx = hx.unsqueeze(dim=1)
        # cat env_feature and action
        x = torch.cat((env_feature, action, hx), dim=1)
        # convolution layer
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # fully connected layer
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # softmax
        x = self.softmax(x)
        return x

    def forward(self, img, txt, action, hx):
        self.Stage1.hx = hx
        Q = self.get_Q(img, txt, action)
        return Q



MEMORY_CAPACITY = 100
N_STATES = 4
GAMMA = 0.95

class DQN():
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        self.eval_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.target_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.num_train = 0

        self.memory = []     # initialize memory
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()

    def store_transition(self, g, f, c, hx, a, r, g_, f_, c_, hx_, batch_size):
        for i in range(batch_size):
            if r[i] == -10000:
                continue
            if r[i] > 100:
                r[i] = 100
            self.memory.append((g[i], f[i], c[i], hx[i], a[i], r[i], g_[i], f_[i], c_[i], hx_[i]))
            if len(self.memory) > MEMORY_CAPACITY:
                self.memory.pop(0)
            self.memory_counter += 1

    def learn(self, batch_size, device=None):
        # target parameter update
        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, batch_size)
        # print("sample_index", sample_index)

        b_g, b_f, b_c, b_hx, b_a, b_r, b_g_, b_f_, b_c_, b_hx_ = self.memory[sample_index[0]]
        b_g = b_g.unsqueeze(dim=0)
        b_f = b_f.unsqueeze(dim=0)
        b_c = b_c.unsqueeze(dim=0)
        b_hx = b_hx.unsqueeze(dim=0)
        b_a = b_a.unsqueeze(dim=0)
        b_r = b_r.unsqueeze(dim=0)
        b_g_ = b_g_.unsqueeze(dim=0)
        b_f_ = b_f_.unsqueeze(dim=0)
        b_c_ = b_c_.unsqueeze(dim=0)
        b_hx_ = b_hx_.unsqueeze(dim=0)
        for i in range(1, len(sample_index)):
            g, f, c, hx, a, r, g_, f_, c_, hx_ = self.memory[sample_index[i]]
            b_g = torch.cat((b_g, g.unsqueeze(dim=0)), dim=0)
            b_f = torch.cat((b_f, f.unsqueeze(dim=0)), dim=0)
            b_c = torch.cat((b_c, c.unsqueeze(dim=0)), dim=0)
            b_hx = torch.cat((b_hx, hx.unsqueeze(dim=0)), dim=0)
            b_a = torch.cat((b_a, a.unsqueeze(dim=0)), dim=0)
            b_r = torch.cat((b_r, r.unsqueeze(dim=0)), dim=0)
            b_g_ = torch.cat((b_g_, g_.unsqueeze(dim=0)), dim=0)
            b_f_ = torch.cat((b_f_, f_.unsqueeze(dim=0)), dim=0)
            b_c_ = torch.cat((b_c_, c_.unsqueeze(dim=0)), dim=0)
            b_hx_ = torch.cat((b_hx_, hx_.unsqueeze(dim=0)), dim=0)
        # print("bhx.shape", b_hx.shape)
        b_a = b_a.unsqueeze(dim=1)
        b_r = b_r.unsqueeze(dim=1)
        #
        # if self.num_train > 5000:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        #     # print("target_net update...")
        #     self.num_train = 0
        # self.num_train += 1
        # print("shape", b_g.shape, b_f.shape, b_c.shape, b_hx.shape, b_a.shape, b_r.shape, b_g_.shape, b_f_.shape, b_c_.shape, b_hx_.shape)
        # # DQN multi
        # q_eval = self.eval_net(b_g, b_f.detach(), b_c, b_hx.detach()).gather(1, b_a)  # shape (batch, 1)
        # q_next = self.target_net(b_g_, b_f_, b_c_, b_hx_).detach()     # detach from graph, don't backpropagate
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)

        # DQN single, retrieval-v5
        b_c1 = torch.zeros(b_c.shape[0], 1, b_c.shape[2]).to(device)
        for i in range(b_a.shape[0]):
            b_c1[i][0] = b_c[i][b_a[i][0]].to(device)
        q_eval = self.eval_net(b_g, b_f.detach(), b_c1, b_hx.detach())  # shape (batch, 1)
        q_next = torch.zeros((batch_size, len(b_c_[0]))).to(device)
        for i in range(len(b_c_[0])):
            q_temp = self.target_net(b_g_, b_f_, b_c_[:, i, :].unsqueeze(1), b_hx_).detach()
            q_next[:, i] = q_temp.squeeze(dim=1)
        flag = 0
        for j in range(len(b_c_)):  # batch_i
            if b_r[j] >= 95:
                q_next[j, :] = torch.zeros((1,len(b_c_[0]))).to(device)
                flag = 1
        q_target = b_r + GAMMA * q_next.max(1)[0].view(batch_size, 1)  # shape (batch, 1)
        # print("q_target", q_eval, q_target, b_r)
        loss = self.loss_func(q_eval.float(), q_target.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class DQN_v1():
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        self.eval_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.target_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.num_train = 0

        self.memory = []     # initialize memory
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()

    def store_transition(self, g, f, c, hx, a, r, g_, f_, c_, hx_, batch_size):
        for i in range(batch_size):
            if r[i] == -10000:
                continue
            self.memory.append((g[i], f[i], c[i], hx[i], a[i], r[i], g_[i], f_[i], c_[i], hx_[i]))
            if len(self.memory) > MEMORY_CAPACITY:
                self.memory.pop(0)
            self.memory_counter += 1

    def learn(self, batch_size, device=None):
        # target parameter update
        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, batch_size)
        # print("sample_index", sample_index)

        b_g, b_f, b_c, b_hx, b_a, b_r, b_g_, b_f_, b_c_, b_hx_ = self.memory[sample_index[0]]
        b_g = b_g.unsqueeze(dim=0)
        b_fl = []
        b_fl.append(b_f)
        # b_f = b_f.unsqueeze(dim=0)
        b_c = b_c.unsqueeze(dim=0)
        b_hx = b_hx.unsqueeze(dim=0)
        b_a = b_a.unsqueeze(dim=0)
        b_r = b_r.unsqueeze(dim=0)
        b_g_ = b_g_.unsqueeze(dim=0)
        b_fl_ = []
        b_fl_.append(b_f_)
        # b_f_ = b_f_.unsqueeze(dim=0)
        b_c_ = b_c_.unsqueeze(dim=0)
        b_hx_ = b_hx_.unsqueeze(dim=0)
        for i in range(1, len(sample_index)):
            g, f, c, hx, a, r, g_, f_, c_, hx_ = self.memory[sample_index[i]]
            b_g = torch.cat((b_g, g.unsqueeze(dim=0)), dim=0)
            b_fl.append(f)
            # b_f = torch.cat((b_f, f.unsqueeze(dim=0)), dim=0)
            b_c = torch.cat((b_c, c.unsqueeze(dim=0)), dim=0)
            b_hx = torch.cat((b_hx, hx.unsqueeze(dim=0)), dim=0)
            b_a = torch.cat((b_a, a.unsqueeze(dim=0)), dim=0)
            b_r = torch.cat((b_r, r.unsqueeze(dim=0)), dim=0)
            b_g_ = torch.cat((b_g_, g_.unsqueeze(dim=0)), dim=0)
            b_fl_.append(f_)
            # b_f_ = torch.cat((b_f_, f_.unsqueeze(dim=0)), dim=0)
            b_c_ = torch.cat((b_c_, c_.unsqueeze(dim=0)), dim=0)
            b_hx_ = torch.cat((b_hx_, hx_.unsqueeze(dim=0)), dim=0)
        # print("bhx.shape", b_hx.shape)
        b_a = b_a.unsqueeze(dim=1)
        b_r = b_r.unsqueeze(dim=1)

        if self.num_train > 5000:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.num_train = 0
        self.num_train += 1
        # DQN multi,
        q_eval = self.eval_net(b_g, b_fl, b_c, b_hx.detach()).gather(1, b_a)  # shape (batch, 1)
        with torch.no_grad():
            q_next = self.target_net(b_g_, b_fl_, b_c_, b_hx_).detach()
        q_next = self.target_net(b_g_, b_fl_, b_c_, b_hx_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)

        loss = self.loss_func(q_eval.float(), q_target.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

# 使用actor critic模型
class DQN_v2():
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        self.eval_net = Retriever(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.target_net = Retriever(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.actor_net = Retriever_v3(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.num_train = 0

        self.memory = []     # initialize memory
        self.memory_counter = 0
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()

    def store_transition(self, g, f, c, hx, a, r, g_, f_, c_, hx_, batch_size):
        for i in range(batch_size):
            if r[i] == -10000:
                continue
            self.memory.append((g[i], f[i], c[i], hx[i], a[i], r[i], g_[i], f_[i], c_[i], hx_[i]))
            if len(self.memory) > MEMORY_CAPACITY:
                self.memory.pop(0)
            self.memory_counter += 1

    def learn(self, batch_size):
        # target parameter update
        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, batch_size)
        # print("sample_index", sample_index)

        b_g, b_f, b_c, b_hx, b_a, b_r, b_g_, b_f_, b_c_, b_hx_ = self.memory[sample_index[0]]
        b_g = b_g.unsqueeze(dim=0)
        b_f = b_f.unsqueeze(dim=0)
        b_c = b_c.unsqueeze(dim=0)
        b_hx = b_hx.unsqueeze(dim=0)
        b_a = b_a.unsqueeze(dim=0)
        b_r = b_r.unsqueeze(dim=0)
        b_g_ = b_g_.unsqueeze(dim=0)
        b_f_ = b_f_.unsqueeze(dim=0)
        b_c_ = b_c_.unsqueeze(dim=0)
        b_hx_ = b_hx_.unsqueeze(dim=0)
        for i in range(1, len(sample_index)):
            g, f, c, hx, a, r, g_, f_, c_, hx_ = self.memory[sample_index[i]]
            b_g = torch.cat((b_g, g.unsqueeze(dim=0)), dim=0)
            b_f = torch.cat((b_f, f.unsqueeze(dim=0)), dim=0)
            b_c = torch.cat((b_c, c.unsqueeze(dim=0)), dim=0)
            b_hx = torch.cat((b_hx, hx.unsqueeze(dim=0)), dim=0)
            b_a = torch.cat((b_a, a.unsqueeze(dim=0)), dim=0)
            b_r = torch.cat((b_r, r.unsqueeze(dim=0)), dim=0)
            b_g_ = torch.cat((b_g_, g_.unsqueeze(dim=0)), dim=0)
            b_f_ = torch.cat((b_f_, f_.unsqueeze(dim=0)), dim=0)
            b_c_ = torch.cat((b_c_, c_.unsqueeze(dim=0)), dim=0)
            b_hx_ = torch.cat((b_hx_, hx_.unsqueeze(dim=0)), dim=0)
        # print("bhx.shape", b_hx.shape)
        b_a = b_a.unsqueeze(dim=1)
        b_r = b_r.unsqueeze(dim=1)

        if self.num_train > 50:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.num_train = 0
        self.num_train += 1
        # print("shape", b_g.shape, b_f.shape, b_c.shape, b_hx.shape, b_a.shape, b_r.shape, b_g_.shape, b_f_.shape, b_c_.shape, b_hx_.shape)
        q_eval = self.eval_net(b_g, b_f.detach(), b_c, b_hx.detach()).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_g_, b_f_, b_c_, b_hx_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)

        critic_loss = torch.mean(F.mse_loss(q_eval.float(), q_target.float()))

        log_probs = self.actor_net(b_g, b_f.detach(), b_c, b_hx.detach())
        delta = q_target - q_eval
        # print("log_probs", log_probs)
        actor_loss = torch.mean(-log_probs * delta.detach()).float()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        # self.target_net.load_state_dict(self.eval_net.state_dict())
        return actor_loss, critic_loss


class DQN_v3():
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device, MULTI_GPU=False, device_ids=None):
        self.eval_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.target_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.actor_net = Retriever_v3(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.num_train = 0
        self.device = device
        self.MULTI_GPU = MULTI_GPU
        self.device_ids = device_ids

        self.memory = []  # initialize memory
        self.memory_counter = 0
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.0001)
        self.critic_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()


    def store_transition(self, g, f, c, hx, a, r, g_, f_, c_, hx_, batch_size, net_mem):
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
                self.memory.append((g_tmp, f_tmp, c_tmp, hx_tmp, a_tmp, reward_temp, g_tmp_, f_tmp_, c_tmp_, hx_tmp_))
                if len(self.memory) > net_mem:
                    self.memory.pop(0)
                self.memory_counter = len(self.memory)
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
        if self.num_train > 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.num_train = 0
        self.num_train += 1
        b_c1 = torch.zeros(b_c.shape[0], 1, b_c.shape[2]).to(device)
        for i in range(b_a.shape[0]):
            b_c1[i][0] = b_c[i][b_a[i][0]].to(device)
        q_eval = self.eval_net(b_g, b_f.detach(), b_c1, b_hx.detach(), b_r)  # shape (batch, 1)
        q_next = torch.zeros((batch_size, len(b_c_[0]))).to(device)
        for i in range(len(b_c_[0])):
            q_temp = self.target_net(b_g_, b_f_, b_c_[:, i, :].unsqueeze(1), b_hx_, b_r).detach()
            q_next[:, i] = q_temp.squeeze(dim=1)
        for j in range(len(b_c_)):
            if b_r[j] >= 900:
                q_next[j, :] = torch.zeros((1, len(b_c_[0]))).to(device)
            if b_r[j] <= -900:
                q_next[j, :] = torch.zeros((1, len(b_c_[0]))).to(device)
        q_target = b_r + 0.995 * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)
        print("q_eval, q_target", q_eval[0], q_target[0], b_r[0], q_eval[0] - q_target[0])
        critic_loss = self.loss_func(q_eval.float(), q_target.float())
        log_probs = self.actor_net(b_g, b_f.detach(), b_c, b_hx.detach(), b_r)
        # print("log_probs", log_probs.shape, log_probs)
        log_probs_temp = torch.zeros((batch_size, 1)).to(device)
        for i in range(b_a.shape[0]):
            log_probs_temp[i][0] = log_probs[i][b_a[i][0]].to(device)
            if log_probs_temp[i][0] > 0.9:
                log_probs_temp[i][0] -= 0.1
        # print("log_probs_temp", log_probs_temp.shape, log_probs_temp)

        delta = q_target - q_eval
        for j in range(len(b_c_)):
            if b_r[j] >= 100:
                delta[j] = b_r[j].to(device)
            if b_r[j] <= -900:
                delta[j] = b_r[j].to(device)
        delta = delta.detach()
        b_r = b_r.detach()
        # print("log_probs_temp", log_probs_temp)
        # print("delta", delta)
        # print("b_r", b_r)
        actor_loss = torch.mean( - torch.log(log_probs_temp) * delta).float()
        # print("actor_loss", actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss, critic_loss


# 打开txt的encode
class A_C_txt():
    def __init__(self, sentence_clip_model2, sentence_clip_preprocess2, device):
        self.eval_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.target_net = Retriever_v5(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.actor_net = Retriever_v3(sentence_clip_model2, sentence_clip_preprocess2, device).to(device)
        self.num_train = 0

        self.memory = []     # initialize memory
        self.memory_counter = 0
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.0001)
        self.critic_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()

    def store_transition(self, g, f, c, hx, a, r, g_, f_, c_, hx_, batch_size, txt_token):
        for i in range(batch_size):
            if r[i] <= -1000:
                continue
            self.memory.append((g[i], f[i], c[i], hx[i], a[i], r[i], g_[i], f_[i], c_[i], hx_[i], txt_token[i]))
            if len(self.memory) > MEMORY_CAPACITY:
                self.memory.pop(0)
            self.memory_counter += 1

    def learn(self, batch_size, device=None):
        # target parameter update
        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, batch_size)
        # print("sample_index", sample_index)

        b_g, b_f, b_c, b_hx, b_a, b_r, b_g_, b_f_, b_c_, b_hx_, txt_token = self.memory[sample_index[0]]
        b_g = b_g.unsqueeze(dim=0)
        b_f = b_f.unsqueeze(dim=0)
        b_c = b_c.unsqueeze(dim=0)
        b_hx = b_hx.unsqueeze(dim=0)
        b_a = b_a.unsqueeze(dim=0)
        b_r = b_r.unsqueeze(dim=0)
        b_g_ = b_g_.unsqueeze(dim=0)
        b_f_ = b_f_.unsqueeze(dim=0)
        b_c_ = b_c_.unsqueeze(dim=0)
        b_hx_ = b_hx_.unsqueeze(dim=0)
        b_txt_token = txt_token.unsqueeze(dim=0)
        for i in range(1, len(sample_index)):
            g, f, c, hx, a, r, g_, f_, c_, hx_ = self.memory[sample_index[i]]
            b_g = torch.cat((b_g, g.unsqueeze(dim=0)), dim=0)
            b_f = torch.cat((b_f, f.unsqueeze(dim=0)), dim=0)
            b_c = torch.cat((b_c, c.unsqueeze(dim=0)), dim=0)
            b_hx = torch.cat((b_hx, hx.unsqueeze(dim=0)), dim=0)
            b_a = torch.cat((b_a, a.unsqueeze(dim=0)), dim=0)
            b_r = torch.cat((b_r, r.unsqueeze(dim=0)), dim=0)
            b_g_ = torch.cat((b_g_, g_.unsqueeze(dim=0)), dim=0)
            b_f_ = torch.cat((b_f_, f_.unsqueeze(dim=0)), dim=0)
            b_c_ = torch.cat((b_c_, c_.unsqueeze(dim=0)), dim=0)
            b_hx_ = torch.cat((b_hx_, hx_.unsqueeze(dim=0)), dim=0)

        # print("bhx.shape", b_hx.shape)
        b_a = b_a.unsqueeze(dim=1)
        # b_r = b_r.unsqueeze(dim=1)

        if self.num_train > 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.num_train = 0
        self.num_train += 1
        # print("shape", b_g.shape, b_f.shape, b_c.shape, b_hx.shape, b_a.shape, b_r.shape, b_g_.shape, b_f_.shape, b_c_.shape, b_hx_.shape)
        b_c1 = torch.zeros(b_c.shape[0], 1, b_c.shape[2]).to(device)
        for i in range(b_a.shape[0]):
            b_c1[i][0] = b_c[i][b_a[i][0]].to(device)
        q_eval = self.eval_net(b_g, b_f.detach(), b_c1, b_hx.detach(), b_r)  # shape (batch, 1)
        q_next = torch.zeros((batch_size, len(b_c_[0]))).to(device)
        for i in range(len(b_c_[0])):
            q_temp = self.target_net(b_g_, b_f_, b_c_[:, i, :].unsqueeze(1), b_hx_, b_r).detach()
            q_next[:, i] = q_temp.squeeze(dim=1)
        for j in range(len(b_c_)):
            if b_r[j] >= 1000:
                q_next[j, :] = torch.zeros((1, len(b_c_[0]))).to(device)
        q_target = b_r + GAMMA * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)
        # print("q_eval, q_target", q_eval, q_target, b_r)
        critic_loss = torch.mean(F.mse_loss(q_eval.float(), q_target.float()))

        # loss = self.loss_func(q_eval.float(), q_target.float())
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        log_probs = self.actor_net(b_g, b_f.detach(), b_c, b_hx.detach(), b_r)
        delta = q_target - q_eval
        # print("log_probs", log_probs)
        actor_loss = torch.mean( - log_probs * delta.detach()).float()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss, critic_loss




