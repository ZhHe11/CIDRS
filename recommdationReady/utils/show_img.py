import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

def show_img(img_id):
    img_path = '/home/data/zh/fur/processed_img/' + str(img_id) + '.jpg'
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.show()


def get_feedback_from_json(G_ids, T_ids, turn, load_dict, sentence_clip_model, G_pre, T_pre):
    feedbacks = []
    g_fbs = []
    t_fbs = []
    batch_size = len(G_ids)
    g_fb = []
    t_fb = []
    for i in range(len(G_ids)):
        G_fbs = load_dict[str(G_ids[i])]
        T_fbs = load_dict[str(T_ids[i])]
        g_objs = list(G_fbs.keys())
        t_objs = list(T_fbs.keys())
        print(T_ids[i])
        for j in range(len(list(G_fbs.keys()))):
            g_fb.append(G_fbs[g_objs[j]].split(",")[0])
            t_fb.append(T_fbs[t_objs[j]].split(",")[0])

    g_fb_token = clip.tokenize(g_fb)     # batch_size*10 * 64
    t_fb_token = clip.tokenize(t_fb)
    img_pre = torch.cat((G_pre, T_pre), dim=0).to(device1)
    txt_token = torch.cat((g_fb_token, t_fb_token), dim=0).to(device1)
    sentence_clip_model.eval()
    sentence_clip_model = sentence_clip_model.to(device1)
    with torch.no_grad():
        image_features, text_features, logit_scale = sentence_clip_model(img_pre, txt_token)
    logits_per_text = logit_scale * text_features @ image_features.t()
    max_index = 0
    for i in range(batch_size):
        b_sim_g_g = logits_per_text[10*i:10*i+10,i]
        b_sim_g_t = logits_per_text[10*i:10*i+10,i + batch_size]
        max_index = torch.argmax(b_sim_g_g - b_sim_g_t)
        g_fb_tmp = g_fb[10*i + max_index]
        b_sim_t_g = logits_per_text[10*(i+batch_size):10*(i+batch_size)+10,i]
        b_sim_t_t = logits_per_text[10*(i+batch_size):10*(i+batch_size)+10,i + batch_size]
        max_index = torch.argmax(b_sim_t_t - b_sim_t_g)
        t_fb_tmp = t_fb[10*i + max_index]
        if turn == -1:
            feedbacks.append("我想要" + t_fb_tmp)
            g_fbs.append("")
            t_fbs.append(t_fb_tmp)
        else:
            feedbacks.append("我不要" + g_fb_tmp + "，我想要" + t_fb_tmp)
            g_fbs.append(g_fb_tmp)
            t_fbs.append(t_fb_tmp)
    # print(G_ids, T_ids)
    return g_fbs, t_fbs, feedbacks


if __name__ == '__main__':

    # img_ids = pd.read_csv('../datasets/test_img_id.csv')
    # print(img_ids.values)
    # for i in img_ids.values:
    #     show_img(i.item())
    #     input()
    # show_img(417478)
    # 413882
    # show_img(413952)

    with open('../dict-test0304.json', 'r') as f:
        load_dict_test = json.load(f)
    wrong_ids = []
    for i in load_dict_test:
        if len(load_dict_test[i]) < 10:
            wrong_ids.append(i)
    test_id = pd.read_csv('../datasets/test_img_id.csv').values
    print(len(test_id))
    test_id_right = []
    count = 0
    for i in test_id:
        i = str(i[0])
        if i in wrong_ids:
            count += 1
            print(count)
            continue
        test_id_right.append(i)


    # save as csv
    print(len(wrong_ids))
    pd.DataFrame(test_id_right).to_csv('../datasets/test_img_id_r.csv', index=False, header=False)
    print(len(test_id_right))





