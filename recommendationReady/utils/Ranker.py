import math
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_image_batch(img_list, batch_ids):
    img_id = img_list[batch_ids]
    return img_id


class Ranker:
    def __init__(self, device):
        self.data_emb = None
        self.device = device
        return

    def update_emb(self, img_list, batch_size, model, sentence_clip_preprocess2):
        num_data = len(img_list)
        self.data_emb = []
        num_batch = math.floor(num_data / batch_size)

        def append_emb(first, last):
            batch_ids = torch.tensor(
                [j for j in range(first, last)], dtype=torch.long)

            image_id = np.array(get_image_batch(img_list, batch_ids))
            img = Image.open(r"E:\data\processed_img" + '/' + str(image_id[0]) + '.jpg')
            imgs = sentence_clip_preprocess2(img).float().cuda().unsqueeze(0)

            time1 = time.time()
            for i in range(1,len(image_id)):
                img = Image.open(r"E:\data\processed_img" + '/' + str(image_id[i]) + '.jpg')
                img = sentence_clip_preprocess2(img).float().cuda().unsqueeze(0)
                imgs = torch.cat((imgs, img))
            print("preprocess2", time.time()-time1)
            with torch.no_grad():
                print(imgs.shape)
                time1 = time.time()
                imgs = model.img_embed(imgs)
                print("model.img_embed time", time.time()-time1)
            self.data_emb.append(imgs)

        for i in range(num_batch):
            append_emb(i*batch_size, (i+1)*batch_size)

        if num_batch * batch_size < num_data:
            append_emb(num_batch * batch_size, num_data)

        self.data_emb = torch.cat(self.data_emb, dim=0)
        print(self.data_emb.shape)
        return

    def nearest_neighbors(self, inputs):
        neighbors = []
        for i in range(inputs.size(0)):
            [_, neighbor] = ((self.data_emb - inputs[i]).pow(2)
                             .sum(dim=1).min(dim=0))

            neighbors.append(neighbor)
        return torch.tensor(neighbors).to(
            device=self.device, dtype=torch.long)

    def compute_rank(self, inputs, target_ids):
        rankings = []
        for i in range(inputs.size(0)):
            distances = (self.data_emb - inputs[i]).pow(2).sum(dim=1)
            ranking = (distances < distances[target_ids[i]]).float().sum(dim=0)
            rankings.append(ranking)
        return torch.tensor(rankings).to(device=self.device, dtype=torch.float)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    filename = r"E:\data\Download\for_data_analysis\img_id.csv"
    img_data = pd.read_csv(filename)
    img_data = img_data['img_id'].values
    # image = get_image_batch(img_data, 0)

    ranker = Ranker('cuda')




