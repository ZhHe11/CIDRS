import math
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import time
import datasets.img_preprocess
from torch.utils.data import dataloader
import matplotlib.pyplot as plt

class Ranker:
    def __init__(self, device, dataset_img, batch_size):
        self.data_emb = None
        self.data_pre = None
        self.data_id = None
        self.device = device
        self.dataset_img = dataset_img
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(dataset_img, batch_size=batch_size, shuffle=False, num_workers=4)

        return

    def update_emb(self, model):
        self.data_emb = []
        self.data_pre = []
        self.data_id = []
        count = 0
        for data in self.dataloader:
            imgs, img_ids = data
            imgs = imgs.to(self.device)
            self.data_pre.append(imgs)
            img_ids = img_ids.to(self.device)
            with torch.no_grad():
                # print(imgs.shape)     # torch.Size([batch_size, 3, 224, 224])
                imgs = model.img_embed(imgs)
            # print("model.img_embed shape", imgs.shape)    # torch.Size([batch_size, 512])
            self.data_emb.append(imgs)
            self.data_id.append(img_ids)
            count += 1
            if count % 10 == 0:
                print("count", count * self.batch_size)
        self.data_emb = torch.cat(self.data_emb, dim=0)
        self.data_pre = torch.cat(self.data_pre, dim=0)
        self.data_id = torch.cat(self.data_id, dim=0)

        # print(self.data_emb.shape)    # torch.Size([num_data(28656), 512])
        return

    def nearest_neighbors(self, inputs):
        neighbors = []
        for i in range(inputs.size(0)):
            [_, neighbor] = ((self.data_emb - inputs[i]).pow(2)
                             .sum(dim=1).min(dim=0))
            neighbors.append(self.data_id[neighbor.item()])
        return torch.tensor(neighbors).to(
            device=self.device, dtype=torch.long)

    def nearest_neighbors_his(self, inputs_his_t, inputs_his_g):
        neighbors = []
        img_pre = []
        distance = torch.zeros(inputs_his_t[0].size(0), self.data_emb.size(0)).to(self.device)
        for inputs in inputs_his_t:
            for batch_i in range(inputs.size(0)):     # batch
                distance_bi = (self.data_emb - inputs[batch_i]).pow(2).sum(dim=1)
                distance[batch_i] += distance_bi
        for inputs in inputs_his_g:
            for batch_i in range(inputs.size(0)):     # batch
                distance_bi = (self.data_emb - inputs[batch_i]).pow(2).sum(dim=1)
                distance[batch_i] -= distance_bi

        for batch_i in range(distance.size(0)):
            [_, neighbor] = distance[batch_i].min(dim=0)
            neighbors.append(self.data_id[neighbor.item()])
        ids = torch.tensor(neighbors).to(device=self.device, dtype=torch.long)
        for id in neighbors:
            id = id.item()
            img_pre.append(self.dataset_img.preprocess(id).to(self.device))
        img_pre = torch.stack(img_pre)
        return ids, img_pre

    def candidate(self, f_embed_t, f_embed_g, k=4, turn=0):
        start_time = time.time()
        neighbors = []
        img_pre = []
        img_candidate = []
        distance = torch.zeros(len(f_embed_t[0]), self.data_emb.size(0)).to(self.device)

        for turn_i in range(len(f_embed_t)):
            for batch_i in range(len(f_embed_t[turn_i])):     # batch
                # dot distance
                distance_bi = torch.mm(f_embed_t[turn_i][batch_i].unsqueeze(0), self.data_emb.t()).squeeze(0)
                distance[batch_i] += distance_bi
                if turn == 0:
                    distance_bi_g = torch.mm(f_embed_g[turn_i][batch_i].unsqueeze(0), self.data_emb.t()).squeeze(0)
                    distance[batch_i] -= 0.5 * distance_bi_g

                # # cos distance
                # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                # distance_bi = -cos(f_embed_t[batch_i], self.data_emb)
                # distance[batch_i] += distance_bi
                # 负向文本
                # distance_bi = cos(f_embed_g[batch_i], self.data_emb)
                # distance[batch_i] += distance_bi
                # # L2 distance
                # distance_bi = (self.data_emb - f_embed[batch_i]).pow(2).sum(dim=1)
            # print("distance time", time.time() - start_time)
        for batch_i in range(distance.size(0)):
            [_, neighbor] = (distance[batch_i]).topk(dim=0, k=k)   # [ 1023, 1022,  1021,  1020]
            # print("neighbor", neighbor)
            data_ids = self.data_id[neighbor]
            # print("data_ids", data_ids)
            neighbors.append(data_ids)
            data_emb = self.data_emb[neighbor]
            data_pre = self.data_pre[neighbor]
            img_candidate.append(data_emb)
            img_pre.append(data_pre)
        id_candidate = torch.stack(neighbors)
        img_candidate = torch.stack(img_candidate)
        img_pre = torch.stack(img_pre)
        return id_candidate, img_candidate, img_pre


    def candidate_fb(self, f_embed_his, f_embed_his_t, f_embed_his_g, k=4, turn=0, batch_size=32):
        start_time = time.time()
        neighbors = []
        img_pre = []
        img_candidate = []
        distance = torch.zeros(batch_size, self.data_emb.size(0)).to(self.device)
        print("f_embed_his", f_embed_his.shape)
        for turn_i in range(f_embed_his.shape[1]):
            for batch_i in range(batch_size):     # batch
                # dot distance
                distance_ = torch.mm(f_embed_his[batch_i][turn_i].unsqueeze(0), self.data_emb.t()).squeeze(0)
                distance_t = torch.mm(f_embed_his_t[batch_i][turn_i].unsqueeze(0), self.data_emb.t()).squeeze(0)
                distance_g = torch.mm(f_embed_his_g[batch_i][turn_i].unsqueeze(0), self.data_emb.t()).squeeze(0)
                distance[batch_i] += 0 * distance_ + 0 * distance_t - 0.5 * distance_g
        for batch_i in range(distance.size(0)):
            [_, neighbor] = (distance[batch_i]).topk(dim=0, k=k)   # [ 1023, 1022,  1021,  1020]
            data_ids = self.data_id[neighbor]
            neighbors.append(data_ids)
            data_emb = self.data_emb[neighbor]
            data_pre = self.data_pre[neighbor]
            img_candidate.append(data_emb)
            img_pre.append(data_pre)
        id_candidate = torch.stack(neighbors)
        img_candidate = torch.stack(img_candidate)
        img_pre = torch.stack(img_pre)
        return id_candidate, img_candidate, img_pre


    def actions_metric(self, f_embed_his, f_embed_his_t, f_embed_his_g, batch_size=32, lens=4):
        # f_embed_his: [turn, batch, emb]
        # distance: [batch, turn, data]
        distance_ = torch.zeros(batch_size, lens, self.data_emb.size(0)).to(self.device)
        distance_t = torch.zeros(batch_size, lens, self.data_emb.size(0)).to(self.device)
        distance_g = torch.zeros(batch_size, lens, self.data_emb.size(0)).to(self.device)
        for turn_i in range(lens):
            for batch_i in range(batch_size):     # batch
                # dot distance
                distance_[batch_i][turn_i] = torch.mm(f_embed_his[batch_i][turn_i].unsqueeze(0), self.data_emb.t()).squeeze(0)
                distance_t[batch_i][turn_i] = torch.mm(f_embed_his_t[batch_i][turn_i].unsqueeze(0), self.data_emb.t()).squeeze(0)
                distance_g[batch_i][turn_i] = torch.mm(f_embed_his_g[batch_i][turn_i].unsqueeze(0), self.data_emb.t()).squeeze(0)
        return distance_, distance_t, distance_g


    def candidate_hx(self, hx, k=4):
        start_time = time.time()
        neighbors = []
        img_pre = []
        img_candidate = []
        distance = torch.zeros(len(hx), self.data_emb.size(0)).to(self.device)

        for batch_i in range(len(hx)):     # batch
            # cos distance
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            distance_bi = -cos(hx[batch_i], self.data_emb)
            distance[batch_i] += distance_bi
            # # L2 distance
            # distance_bi = (self.data_emb - f_embed[batch_i]).pow(2).sum(dim=1)
            # distance[batch_i] += distance_bi
        # print("distance time", time.time() - start_time)
        for batch_i in range(distance.size(0)):
            [_, neighbor] = (-distance[batch_i]).topk(dim=0, k=k)   # [ 1023, 1022,  1021,  1020]
            # print("neighbor", neighbor)
            data_ids = self.data_id[neighbor]
            # print("data_ids", data_ids)
            neighbors.append(data_ids)
            data_emb = self.data_emb[neighbor]
            img_candidate.append(data_emb)
        # print("topk time", time.time() - start_time)
        id_candidate = torch.stack(neighbors)
        # print("id_candidate", id_candidate)
        img_candidate = torch.stack(img_candidate)
        # print("img_candidate", img_candidate.shape)
        # for ids in neighbors:
        #     img_pre_b = []
        #     for id in ids:
        #         id = id.item()
        #         img_pre_b.append(self.dataset_img.preprocess(id).to(self.device))
        #     img_pre_b = torch.stack(img_pre_b)
        #     img_pre.append(img_pre_b)
        # img_pre = torch.stack(img_pre)
        # print("preprocess time", time.time() - start_time)
        return id_candidate, img_candidate, img_pre

    def candidate_his(self, f_embed_his, k=1):
        neighbors = []
        img_pre = []
        img_candidate = []
        distance = torch.zeros(len(f_embed_his[0]), self.data_emb.size(0)).to(self.device)
        for turn_i in range(len(f_embed_his)):
            for batch_i in range(len(f_embed_his[turn_i])):     # batch
                # cos distance
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                distance_bi = cos(f_embed_his[turn_i][batch_i], self.data_emb)
                distance[batch_i] += distance_bi

        for batch_i in range(distance.size(0)):
            [_, neighbor] = (distance[batch_i]).topk(dim=0, k=k)   # [ 1023, 1022,  1021,  1020]
            data_ids = self.data_id[neighbor]
            neighbors.append(data_ids)
            data_emb = self.data_emb[neighbor]
            img_candidate.append(data_emb)

        id_candidate = torch.stack(neighbors)
        # print("id_candidate", id_candidate)
        img_candidate = torch.stack(img_candidate)
        # print("img_candidate", img_candidate.shape)
        return id_candidate, img_candidate



    def get_reward(self, G_next_imgs, T_imgs):
        # print("G_next_imgs", G_next_imgs.shape)
        # print("T_imgs", T_imgs.shape)
        # distance = torch.zeros(self.data_emb.size(0)).to(self.device)
        # print("distance", distance.shape)
        # # cos distance
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # distance = cos(self.data_emb, T_imgs.detach())
        # # distance = (self.data_emb - G_next_imgs).pow(2).sum(dim=1)
        # print("distance", distance)
        # # statistical distribution
        # x = np.linspace(1,len(distance),len(distance))
        # plt.plot(x, distance.cpu().numpy())
        # plt.show()
        # exit()
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        # print(G_next_imgs.shape)
        # print(T_imgs.shape)
        d = cos(G_next_imgs, T_imgs)
        # # L2
        # d = (G_next_imgs - T_imgs).pow(2).sum(dim=0)  # 200 175 150 125
        if d < 0.3:
            reward = -2
        elif d < 0.35:
            reward = -1
        elif d < 0.45:
            reward = 0
        elif d < 0.5:
            reward = 1
        else:
            reward = 2
        return reward


    def compute_rank(self, inputs, targets, distance='dot'):
        if distance == 'dot':
            rankings = []
            # print("inputs", inputs.shape)
            for i in range(inputs.size(0)):
                # print("inputs[i]", inputs[i].shape)
                distances = torch.mm(inputs[i].unsqueeze(0), self.data_emb.t()).squeeze(0)
                # print("distances", distances)
                target_distance = torch.mm(inputs[i].unsqueeze(0), targets[i].unsqueeze(0).t()).squeeze(0)
                # print("target_distance", target_distance)
                ranking = (distances > target_distance).float().sum(dim=0)
                # print("ranking", ranking)
                rankings.append(ranking)
            return torch.tensor(rankings).to(device=self.device, dtype=torch.float)

        if distance == 'L2':
            rankings = []
            for i in range(inputs.size(0)):
                distances = (self.data_emb - inputs[i]).pow(2).sum(dim=1)
                target_distance = (inputs[i] - targets[i]).pow(2).sum()
                ranking = (distances < target_distance).float().sum(dim=0)
                rankings.append(ranking)
            return torch.tensor(rankings).to(device=self.device, dtype=torch.float)
        if distance == 'cos':
            rankings = []
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            for i in range(inputs.size(0)):
                distances = cos(inputs[i], self.data_emb)
                target_distance = cos(inputs[i], targets[i])
                ranking = (distances > target_distance).float().sum(dim=0)
                rankings.append(ranking)
            return torch.tensor(rankings).to(device=self.device, dtype=torch.float)
        if distance == 'cos1':
            rankings = []
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            distances = cos(inputs, self.data_emb)
            # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            # print("inputs", inputs.shape)
            # print("targets", targets.shape)
            target_distance = cos(inputs, targets)
            ranking = (distances > target_distance).float().sum(dim=0)
            rankings.append(ranking)
            return torch.tensor(rankings).to(device=self.device, dtype=torch.float)
# if __name__ == '__main__':
#     pass