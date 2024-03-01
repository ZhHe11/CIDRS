import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel


class AttDes(nn.Module):
    def __init__(self, args):
        super(AttDes, self).__init__()
        hidden_dim = args.AD_hidden_dim
        





