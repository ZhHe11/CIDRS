import torch
import torch.nn as nn
from cn_clip.clip import load_from_name, available_models
import cn_clip.clip as clip
from PIL import Image
from torch.autograd import Variable


class Stage1(nn.Module):
    def __init__(self, hid_dim):
        super(Stage1, self).__init__()
        self.layer1 = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)
        

    def forward(self, img, txt, reward):
        img = self.linear_1(img)
        txt = self.linear_2(txt)
        t_embed = torch.cat((img, txt), dim=1)
        t_embed = self.gru(t_embed)
        t_embed = self.norm(t_embed)
        return t_embed

    def gru(self, t_embed):
        t_embed = self.linear_4(self.linear_3(t_embed))
        t_embed = self.norm(t_embed)
        self.hx = self.rnn(t_embed, self.hx)
        self.hx = self.norm(self.hx)
        x = self.linear_4(self.hx)
        return x

    def init_hid(self, batch_size=1):
        self.hx = Variable(torch.Tensor(batch_size, self.hid_dim).zero_())
        return

    def detach_hid(self):
        self.hx = Variable(self.hx.data)
        return







