import argparse
import datetime
import json
import random
import time
import math
import os

import numpy as np
from pathlib import Path

import torch
from nltk.translate import bleu_score

import dataset.data_loader
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
from models import prefixLM, tokenizer
import nltk
import jieba
# from engine import train_one_epoch, validate
#
# import utils.misc as utils
from models import __init__
# from dataset import build_dataset
# from engine import train_one_epoch, validate_txt

from einops import rearrange
from pytorch_pretrained_bert.tokenization import BertTokenizer

def get_args_parser():
    parser = argparse.ArgumentParser('Set parser', add_help=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu_id', default='0', type=str)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default=r'E:\data\Download\fur\dataset\data_for_test2.csv')
    parser.add_argument('--dataset_name', type=str, default='Furniture')
    parser.add_argument('--img_root', type=str, default=r'E:\data\pictures')
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

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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


def validate(img1_id, img2_id, obj, model_path):
    parser = argparse.ArgumentParser('AttDes training script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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

def validate_one_img(model, dataset_all, img_id, obj, device, tokenizer):
    # print("start validate...")
    start_time = time.time()
    end1_time = time.time()
    model.eval()
    print(obj)
    img_data, des, obj_data, target, img_id, obj_given = dataset_all.get_all_from_id(img_id, obj)
    print(obj_given)
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
    logits = model.to_logits(out)[:, -1]
    sample = torch.argmax(logits, dim=-1)
    value, index = logits.topk(3, dim=-1)
    sample = index[0][0].unsqueeze(0)
    sample_2nd = index[0][1].unsqueeze(0)
    sample_3rd = index[0][2].unsqueeze(0)
    tgt_txt_2nd = tgt_txt
    tgt_txt_3rd = tgt_txt

    cur_len = 1
    while (cur_len < model.tgt_len and sample != 102):  # 102 is the id of [SEP]
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
    label_txt = []
    output_txt = []
    obj_txt = []
    for token in des[0].tolist():
        if token > 103:
            label_txt.append(token)
    for token in tgt_txt[0].tolist():
        if token > 103:
            output_txt.append(token)
    # for token in obj_data[0].tolist():
    #     if token > 103:
    #         obj_txt.append(token)
    label_txt = tokenizer.convert_ids_to_tokens(label_txt)
    label_txt = ''.join(label_txt)

    # obj_txt = tokenizer.convert_ids_to_tokens(obj_txt)
    output_txt = tokenizer.convert_ids_to_tokens(output_txt)
    output1 = ''.join(output_txt)

    # 2nd
    cur_len = 1
    while (cur_len < model.tgt_len and sample_2nd != 102):  # 102 is the id of [SEP]
        tgt_txt_2nd = torch.cat((tgt_txt_2nd, sample_2nd.unsqueeze(1)), dim=-1)
        tgt_txt_embed = model.txt_embed(tgt_txt_2nd)
        cur_len += 1
        tgt_txt_embed += model.txt_pos_embed(torch.arange(cur_len, device=device))
        # out = model.transformer(prefix, tgt_txt_embed)
        out = model.ModelOne(q=obj_embed, k=img_emed, v=img_emed,
                             tgt_embeded=tgt_txt_embed, des_embed=des_embed, obj_embed=obj_embed, img_embed=img_emed,
                             tgt_mask=None)
        logits = model.to_logits(out)[:, -1]
        # logits = logits[:, :-26]
        # print(logits)
        sample_2nd = torch.argmax(logits, dim=-1)

    output_txt = []
    for token in tgt_txt_2nd[0].tolist():
        if token > 103:
            output_txt.append(token)
    output_txt = tokenizer.convert_ids_to_tokens(output_txt)
    output2 = ''.join(output_txt)
    # 3rd
    cur_len = 1
    while (cur_len < model.tgt_len and sample_3rd != 102):  # 102 is the id of [SEP]
        tgt_txt_3rd = torch.cat((tgt_txt_3rd, sample_3rd.unsqueeze(1)), dim=-1)
        tgt_txt_embed = model.txt_embed(tgt_txt_3rd)
        cur_len += 1
        tgt_txt_embed += model.txt_pos_embed(torch.arange(cur_len, device=device))
        # out = model.transformer(prefix, tgt_txt_embed)
        out = model.ModelOne(q=obj_embed, k=img_emed, v=img_emed,
                             tgt_embeded=tgt_txt_embed, des_embed=des_embed, obj_embed=obj_embed, img_embed=img_emed,
                             tgt_mask=None)
        logits = model.to_logits(out)[:, -1]
        # logits = logits[:, :-26]
        sample_3rd = torch.argmax(logits, dim=-1)

    output_txt = []
    for token in tgt_txt_3rd[0].tolist():
        if token > 103:
            output_txt.append(token)
    output_txt = tokenizer.convert_ids_to_tokens(output_txt)
    output3 = ''.join(output_txt)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(output1)
    print(output2)
    print(output3)
    print('Validate time {}'.format(total_time_str))
    return label_txt, output1, output2, output3


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # parser = argparse.ArgumentParser('AttDes training script', parents=[get_args_parser()])
    # args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # main(args)
    model_name = '005'
    model_path = r'E:\data\Download\models\attribute_desciption\outputs' + '/' + model_name + '/' + 'checkpoint0019.pth'
    objs = ["空间","客厅","卧室","墙面","餐厅","公寓","住宅","沙发","家具","地毯","厨房","书房","背景墙","吊灯","墙",
           "卫生间","儿童","床品","装饰","壁纸","地板","窗帘","吊顶","餐椅","别墅","地面","结构","布艺","餐桌","画"]
    for obj in objs:
        print(obj)
        out = validate('550695', '550567', obj, model_path)
        sentences1 = out[0][0].replace('；', '，').split('，')
    # gt = ""
    #
    # for i in sentences1:
    #     if obj in i:
    #         gt = i
    # gt = " ".join(jieba.cut(gt))
    # print(gt)
    # for i in out[0]:
    #     i = " ".join(jieba.cut(i))
    #     print(i)
    #     print(gt)
    #     bleu = nltk.translate.bleu_score.sentence_bleu([i], gt)
    #     print(bleu)







