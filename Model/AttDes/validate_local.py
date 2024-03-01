import argparse
import datetime
import json
import random
import time
import math
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from nltk.translate import bleu_score
import sys
sys.path.append(r"E:\data\streamlit\Model\AttDes")
sys.path.append(r"E:\data\streamlit\Model\CLIP")
from AttDes import dataset
from AttDes.dataset import data_loader
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import AttDes.models as models
from AttDes.models import prefixLM, tokenizer

import nltk
import jieba
# from engine import train_one_epoch, validate
#
# import utils.misc as utils
# from models import __init__
# from dataset import build_dataset
# from engine import train_one_epoch, validate_txt

from einops import rearrange
from pytorch_pretrained_bert.tokenization import BertTokenizer

def get_args_parser():
    parser = argparse.ArgumentParser('Set parser', add_help=False)
    parser.add_argument('--device', default='cuda')
    # parser.add_argument('--gpu_id', default='0', type=str)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='/home/data/zh/project1/DatasetProcess/rawData/data_for_test2.csv')
    parser.add_argument('--dataset_name', type=str, default='Furniture')
    parser.add_argument('--img_root', type=str, default='/home/data/zh/fur/processed_img')
    parser.add_argument('--output_dir', default='./outputs/validate', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--resume', default='', help='resume for checkpoint')
    parser.add_argument('--bert_model', default='bert-base-chinese', type=str)
    parser.add_argument('--des_len', default=256, type=int)
    parser.add_argument('--obj_len', default=8, type=int)
    parser.add_argument('--tgt_len', default=35, type=int)


    # Train parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    # Model parameters
    parser.add_argument('--AD_hidden_dim', default=256, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    # visual_model parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")

    return parser


def main(args):
    device = torch.device(args.device)

    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    normalize = transforms.Normalize(mean=[0.5024, 0.4993, 0.4992],
                                     std=[0.1673, 0.1695, 0.1705])
    the_transforms = transforms.Compose([transforms.Resize((448, 448)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                        ])
    dataset_all = AttDes.dataset.data_loader.AttDesDataset(args.data_root, args.dataset_name,
                                                      des_len=args.des_len,
                                                      obj_len=args.obj_len,
                                                      tgt_len=args.tgt_len,
                                                      img_root=args.img_root,
                                                      transform=the_transforms)

    dataloader_val = DataLoader(dataset_all,
                                  batch_size=args.batch_size,
                                  shuffle=False)
    print("data loaded...")

    Tokenizer = tokenizer.ChineseTokenizer()
    PrefixLM_configure = dict(d_model=args.d_model, des_len=args.des_len, obj_len=args.obj_len, tgt_len=args.tgt_len,
                              input_resolution=448,
                              patch_size=16,
                              num_text_tokens=20000,
                              txt_seq_len=10000,
                              heads=4,
                              enc_depth=8,
                              dec_depth=8,
                              d_ff=1024,
                              dropout=0.1)
    model = prefixLM.PrefixLM(**PrefixLM_configure).to(device)
    model.load_state_dict(torch.load('./outputs/005/checkpoint0019.pth'))

    output_dir = Path(args.output_dir)
    with (output_dir / "log.txt").open("a") as f:
        f.write(str(args) + "\n")

    print("start validate...")
    start_time = time.time()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    for epoch in range(args.start_epoch, args.epochs):
        validate_txt(args, model, dataloader_val, device, batch_size=args.batch_size)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Validate time {}'.format(total_time_str))


def load_AttDes_Model(model_path, device):
    parser = argparse.ArgumentParser('AttDes training script', parents=[get_args_parser()])
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.5024, 0.4993, 0.4992],
                                     std=[0.1673, 0.1695, 0.1705])
    the_transforms = transforms.Compose([transforms.Resize((448, 448)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                        ])
    dataset_all = data_loader.AttDesDataset(args.data_root, args.dataset_name,
                                                      des_len=args.des_len,
                                                      obj_len=args.obj_len,
                                                      tgt_len=args.tgt_len,
                                                      img_root=args.img_root,
                                                      transform=the_transforms)
    PrefixLM_configure = dict(d_model=args.d_model, des_len=args.des_len, obj_len=args.obj_len, tgt_len=args.tgt_len,
                              input_resolution=448,
                              patch_size=16,
                              num_text_tokens=20000,
                              txt_seq_len=10000,
                              heads=4,
                              enc_depth=8,
                              dec_depth=8,
                              d_ff=1024,
                              dropout=0.1)
    time_1 = time.time()
    model = prefixLM.PrefixLM(**PrefixLM_configure).to(device)
    model.load_state_dict(torch.load(model_path))
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    time_2 = time.time()
    print('Load model takes {}s'.format(time_2 - time_1))
    return model, dataset_all, tokenizer



def validate(img1_id, img2_id, obj, model_path):
    parser = argparse.ArgumentParser('AttDes training script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    #
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    normalize = transforms.Normalize(mean=[0.5024, 0.4993, 0.4992],
                                     std=[0.1673, 0.1695, 0.1705])

    the_transforms = transforms.Compose([transforms.Resize((448, 448)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                        ])
    dataset_all = dataset.data_loader.AttDesDataset(args.data_root, args.dataset_name,
                                                      des_len=args.des_len,
                                                      obj_len=args.obj_len,
                                                      tgt_len=args.tgt_len,
                                                      img_root=args.img_root,
                                                      transform=the_transforms)
    PrefixLM_configure = dict(d_model=args.d_model, des_len=args.des_len, obj_len=args.obj_len, tgt_len=args.tgt_len,
                              input_resolution=448,
                              patch_size=16,
                              num_text_tokens=20000,
                              txt_seq_len=10000,
                              heads=4,
                              enc_depth=8,
                              dec_depth=8,
                              d_ff=1024,
                              dropout=0.1)
    time_1 = time.time()
    model = prefixLM.PrefixLM(**PrefixLM_configure).to(device)
    model.load_state_dict(torch.load(model_path))
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    time_2 = time.time()
    print('Load model takes {}s'.format(time_2 - time_1))
    out_list = []

    label_txt, output1, output2, output3 = validate_one_img(model, dataset_all, img1_id, obj, device, tokenizer)
    out_list.append([label_txt, output1, output2, output3])
    label_txt, output1, output2, output3 = validate_one_img(model, dataset_all, img2_id, obj, device, tokenizer)
    out_list.append([label_txt, output1, output2, output3])
    return out_list

def get_data_from_csv_by_id(path, id):
    data_csv = pd.read_csv(path, encoding='utf-8')
    # print(data_csv)
    pic_id_list = data_csv['pic_id'].values
    des_list = data_csv['des'].values

    for i in range(len(pic_id_list)):
        if str(pic_id_list[i]) == str(id):
            return des_list[i]

    return ""

def validate_one_img(model, dataset_all, img_ids, obj_given, device, tokenizer):
    batch_size = len(img_ids)
    start_time = time.time()
    model.eval()
    imgs = []
    dess = []
    objs = []
    for i in range(len(img_ids)):
        img, des, obj = dataset_all.get_all_from_id(img_ids[i], obj_given[i])
        # print("get img from id time:", time.time() - start_time)  # 3s
        imgs.append(img)
        dess.append(des)
        objs.append(obj)
    img_data = torch.stack(imgs).to(device)
    des_data = torch.stack(dess).to(device)
    obj_data = torch.stack(objs).to(device)
    # print("get batch time:", time.time() - start_time) # 3s
    img_emed = model.ResNet(img_data)
    img_emed = rearrange(img_emed, 'b c h w -> b (h w) c')
    img_emed += model.img_pos_embed(img_emed)

    des_embed = model.txt_embed(des_data)
    des_embed += model.txt_pos_embed(torch.arange(model.des_len, device=device))
    obj_embed = model.txt_embed(obj_data)
    obj_embed = obj_embed + model.txt_pos_embed(torch.arange(model.obj_len, device=device))


    tgt_txt = torch.zeros(batch_size, 1, dtype=torch.long, device=device) + 101
    tgt_txt_embed = model.txt_embed(tgt_txt)
    tgt_txt_embed += model.txt_pos_embed(torch.arange(1, device=device) + model.tgt_len)

    # M_005
    out = model.ModelOne(q=obj_embed, k=img_emed, v=img_emed,
                         tgt_embeded=tgt_txt_embed, des_embed=des_embed, obj_embed=obj_embed, img_embed=img_emed,
                         tgt_mask=None)
    logits = model.to_logits(out)[:, -1]
    _, index = logits.topk(3, dim=-1)
    # value: tensor([[7.3227, 7.2289, 6.4169],
    #               [9.6868, 7.0598, 6.3911]], device='cuda:0', grad_fn= < TopkBackward0 >)
    # index: tensor([[4677, 2199, 2647],
    #                [4510, 3763, 2145]], device='cuda:0')
    sample_1st = index[:,0]
    sample_2nd = index[:,1]
    sample_3rd = index[:,2]
    tgt_txt0 = tgt_txt
    output_list = []
    # print("get 1,2,3 sample time:", time.time() - start_time) # 0.01s
    for sample in [sample_1st, sample_2nd, sample_3rd]:
        tgt_txt = tgt_txt0
        cur_len = 1
        while (cur_len < model.tgt_len and sample.max() != 102):  # 102 is the id of [SEP]
            tgt_txt = torch.cat((tgt_txt, sample.unsqueeze(1)), dim=-1)
            tgt_txt_embed = model.txt_embed(tgt_txt)
            cur_len += 1
            tgt_txt_embed += model.txt_pos_embed(torch.arange(cur_len, device=device))
            # out = model.transformer(prefix, tgt_txt_embed)
            out = model.ModelOne(q=obj_embed, k=img_emed, v=img_emed,
                                 tgt_embeded=tgt_txt_embed, des_embed=des_embed, obj_embed=obj_embed, img_embed=img_emed,
                                 tgt_mask=None)
            logits = model.to_logits(out)[:, -1]
            sample = torch.argmax(logits, dim=-1)
        # print("one batch sentence token time:", time.time() - start_time) # 0.6s
        output_1 = []
        for i in range(batch_size):
            output_txt = []
            for token in tgt_txt[i].tolist():
                if token > 103:
                    output_txt.append(token)
            output_txt = tokenizer.convert_ids_to_tokens(output_txt)
            output_txt = ''.join(output_txt)
            output_1.append(output_txt[1:])
        output_list.append(output_1)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Validate time {}'.format(total_time_str))
    # print(output_list)
    return output_list


def generate_texts(img_id, obj, model_path):
    parser = argparse.ArgumentParser('AttDes training script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    normalize = transforms.Normalize(mean=[0.5024, 0.4993, 0.4992],
                                     std=[0.1673, 0.1695, 0.1705])
    the_transforms = transforms.Compose([transforms.Resize((448, 448)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                         ])
    dataset_all = dataset.data_loader.AttDesDataset(args.data_root, args.dataset_name,
                                                    des_len=args.des_len,
                                                    obj_len=args.obj_len,
                                                    tgt_len=args.tgt_len,
                                                    img_root=args.img_root,
                                                    transform=the_transforms)
    PrefixLM_configure = dict(d_model=args.d_model, des_len=args.des_len, obj_len=args.obj_len, tgt_len=args.tgt_len,
                              input_resolution=448,
                              patch_size=16,
                              num_text_tokens=20000,
                              txt_seq_len=10000,
                              heads=4,
                              enc_depth=8,
                              dec_depth=8,
                              d_ff=1024,
                              dropout=0.1)
    time_1 = time.time()
    model = prefixLM.PrefixLM(**PrefixLM_configure).to(device)
    model.load_state_dict(torch.load(model_path))
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    time_2 = time.time()
    print('Load model takes {}s'.format(time_2 - time_1))

    print("start generate_texts")
    start_time = time.time()
    end1_time = time.time()
    model.eval()
    img_data, des, obj_data, target, img_id, obj_given = dataset_all.get_all_from_id(img_id, obj)

    img_data = img_data.unsqueeze(0).to(device)
    des = des.unsqueeze(0).to(device)
    obj_given = obj_given.unsqueeze(0).to(device)
    label = target.unsqueeze(0).to(device)

    img_emed = model.ResNet(img_data)

    img_emed = rearrange(img_emed, 'b c h w -> b (h w) c')
    img_emed += model.img_pos_embed(img_emed)

    des_embed = model.txt_embed(des)
    des_embed += model.txt_pos_embed(torch.arange(model.des_len, device=device))
    obj_embed = model.txt_embed(obj_given)
    obj_embed = obj_embed + model.txt_pos_embed(torch.arange(model.obj_len, device=device))
    tgt_txt = torch.zeros(1, 1, dtype=torch.long, device=device) + 101
    tgt_txt_embed = model.txt_embed(tgt_txt)
    tgt_txt_embed += model.txt_pos_embed(torch.arange(1, device=device) + model.tgt_len)

    # M_005
    out = model.ModelOne(q=obj_embed, k=img_emed, v=img_emed,
                         tgt_embeded=tgt_txt_embed, des_embed=des_embed, obj_embed=obj_embed, img_embed=img_emed,
                         tgt_mask=None)




if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # parser = argparse.ArgumentParser('AttDes training script', parents=[get_args_parser()])
    # args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # main(args)
    model_name = '005'
    model_path = r'E:\data\Download\models\attribute_desciption\outputs' + '/' + model_name + '/' + 'checkpoint0019.pth'
    obj = ["空间","客厅","卧室","墙面","餐厅","公寓","住宅","沙发","家具","地毯","厨房","书房","背景墙","吊灯","墙",
           "卫生间","儿童","床品","装饰","壁纸","地板","窗帘","吊顶","餐椅","别墅","地面","结构","布艺","餐桌","画"]

    out = generate_texts('550695', obj, model_path)












