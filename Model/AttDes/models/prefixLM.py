'''
   author: yulong-XJTU
'''
import torch
from torch import nn
import torch.nn.functional as F
import copy
from AttDes.models.transformer import Transformer, subsequent_mask, ModelOne, Model005, Model006
from axial_positional_embedding import AxialPositionalEmbedding
from AttDes.models.resblock import BottleneckBlock
from random import randint
from einops import rearrange

def clone(module,N):
    '''copy the given module N times'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PrefixLM(nn.Module):
    def __init__(
            self, des_len, obj_len, tgt_len,
            d_model=512,
            input_resolution=224,
            patch_size=16,
            num_text_tokens=10000,
            txt_seq_len=256,
            prefix_txt_len=25,
            target_txt_len=52,
            max_trunc_txt_len=15,
            heads=8,
            enc_depth=12,
            dec_depth=12,
            d_ff=1024,
            dropout=0.,
    ):
        super(PrefixLM,self).__init__()
        assert input_resolution % patch_size==0 and max_trunc_txt_len<=prefix_txt_len and max_trunc_txt_len<txt_seq_len
        self.ResNet = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=64, kernel_size=patch_size, stride=patch_size, bias=True),
                                    BottleneckBlock(in_channels=64,out_channels=256,bottleneck_channels=64,),
                                    BottleneckBlock(in_channels=256,out_channels=d_model,bottleneck_channels=128)])
        self.des_len = des_len
        self.obj_len = obj_len
        self.tgt_len = tgt_len
        self.txt_embed = nn.Embedding(num_text_tokens, d_model, padding_idx=0)
        self.txt_pos_embed = nn.Embedding(self.des_len,d_model)
        image_fmap_size = input_resolution // patch_size    # 448 // 16
        self.img_tokens_len=image_fmap_size ** 2
        # self.img_pos_embed=nn.Embedding(self.img_tokens_len,d_model)
        self.img_pos_embed = AxialPositionalEmbedding(d_model, axial_shape=(image_fmap_size, image_fmap_size))
        self.txt_seq_len = txt_seq_len
        self.target_txt_len = target_txt_len
        self.prefix_txt_len = prefix_txt_len

        self.max_trunc_txt_len=max_trunc_txt_len
        self.num_text_tokens = num_text_tokens
        self.dim_embed=d_model
        self.input_resolution=input_resolution
        self.patch_size=patch_size
        # self.temperature = nn.Parameter(torch.tensor(1.))       # 论文中没提到
        self.transformer=Transformer(d_model,heads,enc_depth,dec_depth,d_ff,dropout=dropout)
        self.ModelOne = Model005(d_model,heads,enc_depth,dec_depth,d_ff,dropout=dropout)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.num_text_tokens)
        )

    def forward(self, img, des, obj, tgt, return_loss=False):
        device = des.device
        n = des.shape[0]
        img_emed = self.ResNet(img)
        img_emed = rearrange(img_emed,'b c h w -> b (h w) c')
        img_emed = img_emed + self.img_pos_embed(img_emed)
        del img
        #add<CLS>, if you change the tokenizer, don't forget  to change the token ID. another [SEP] token is added at the ending(in the tokenizer.py,please check.)
        tgt = F.pad(tgt, (1, 0), value=4)
        labels = tgt[:,1:]
        tgt = tgt[:,:-1]
        # print('des:', torch.min(des), torch.max(des))
        des_embed = self.txt_embed(des)
        des_embed = des_embed + self.txt_pos_embed(torch.arange(self.des_len, device=device))

        obj_embed = self.txt_embed(obj)
        obj_embed = obj_embed + self.txt_pos_embed(torch.arange(self.obj_len, device=device))

        tgt_embed = self.txt_embed(tgt)
        tgt_embed = tgt_embed + self.txt_pos_embed(torch.arange(self.tgt_len, device=device))
        tgt_mask = subsequent_mask(self.tgt_len).to(device)

        # baseline
        # prefix = torch.cat((img_emed, des_embed, obj_embed), dim=1)
        # tgt_mask = subsequent_mask(self.tgt_len).to(device)
        # out = self.transformer(prefix, tgt_embed, tgt_mask=tgt_mask)

        # ModelOne

        out = Model005(q=obj_embed, k=img_emed, v=img_emed,
                            tgt_embeded=tgt_embed, des_embed=des_embed, obj_embed=obj_embed, img_embed=img_emed,
                            tgt_mask=tgt_mask)

        logits = self.to_logits(out)
        return logits, labels
        # if not return_loss:
        #     return logits
        # # temp = self.temperature.exp()
        # logits = rearrange(logits, 'b n c -> b c n')
        # # logits=logits*temp #带温度参数
        # loss=F.cross_entropy(logits,labels,ignore_index=0)
        # return loss

