import torch
import torch.nn as nn
from cn_clip.clip import load_from_name, available_models
import cn_clip.clip as clip
from PIL import Image
from torch.autograd import Variable

class Stage1(nn.Module):
    def __init__(self, hid_dim):
        super(Stage1, self).__init__()
        self.hid_dim = hid_dim
        self.linear_1 = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)
        self.linear_2 = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)
        self.linear_3 = nn.Linear(in_features=hid_dim * 2, out_features=hid_dim, bias=False)
        self.rnn = nn.GRUCell(hid_dim, hid_dim, bias=False)
        self.linear_4 = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)

    def forward(self, img, txt):
        img = self.linear_1(img)
        txt = self.linear_2(txt)

    def gru(self, t_embed):
        t_embed = self.linear_3(t_embed)
        self.hx = self.rnn(t_embed, self.hx)
        x = self.linear_4(self.hx)
        return x

    def init_hid(self, batch_size=1):
        self.hx = Variable(torch.Tensor(batch_size, self.hid_dim).zero_())
        return

    def detach_hid(self):
        self.hx = Variable(self.hx.data)
        return


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


device = torch.device('cuda')
model_name = '005/epoch_16.pt'
resume = r'E:\data\project2\Chinese-CLIP-master\Chinese-CLIP-master\usage\save_path' + '/' + model_name
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='../../data/pretrained_weights/',
                                   resume=resume)
convert_models_to_fp32(model)

Given_path = r'E:\data\pictures' + '/' + str(550567) + '.jpg'

Given = preprocess(Image.open(Given_path)).unsqueeze(0).to(device)
Given_embed = model.encode_image(Given)

feedback = "我不要搭配灰色窗帘，我想要搭配动物图案的窗帘"
fb = clip.tokenize(feedback).to(device)
fb_embed = model.encode_text(fb)

# print(Given_embed)
print(Given_embed.shape)
# print(fb_embed)
print(fb_embed.shape)

t_embed = torch.cat((Given_embed, fb_embed), dim=1)
print(t_embed.shape)

GRU = Stage1(hid_dim=512).to(device)
GRU.train()
GRU.init_hid()
GRU.hx = GRU.hx.to(device)
t_embed = GRU.gru(t_embed=t_embed)
print(t_embed.shape)






