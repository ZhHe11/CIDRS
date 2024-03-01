import sys
sys.path.append(r"E:\data\streamlit")
sys.path.append(r"E:\data\streamlit\Model")
sys.path.append(r"E:\data\streamlit\Model\AttDes")
sys.path.append(r"E:\data\streamlit\Model\CLIP")
import streamlit as st
from PIL import Image
# import models.AttDes.validate_local
import AttDes.validate_local
import streamlit.components.v1 as components
import CLIP.usage.calculate


# st.set_page_config(layout="wide")
st.title('Feed_back V1')

col1, col2 = st.columns(2)
text1 = '550567'
text1 = col1.text_input("example: 550567", text1)
text2 = '550695'
text2 = col2.text_input("example: 550695", text2)
# img1 = Image.open(r'E:\data\pictures\\' + text1 + '.jpg')
# st.image(img1, caption='img1')

with st.container():
    col1, col2 = st.columns([1, 1])
    if text1 != '':
        img1_id = text1
        img1_path = r'E:\data\pictures' + '/' + text1 + '.jpg'
        img1 = Image.open(img1_path).resize((224,224))
        col1.image(img1, caption='img_given', use_column_width='auto')

    if text2 != '':
        img2_id = text2
        img2_path = r'E:\data\pictures' + '/' + text2 + '.jpg'
        img2 = Image.open(img2_path).resize((224,224))
        col2.image(img2, caption='img_target', use_column_width='auto')
st.write("")
out_list = []
is_run = 0
with st.container():
    col1, col2, col3 = st.columns([1,1,1])

    model_name = '011'
    model_name = col1.text_input('obj_des model_name', model_name)
    model_path = r'E:\data\Download\models\attribute_desciption\outputs' + '/' + model_name + '/' + 'checkpoint0019.pth'
    obj = ''
    obj = col2.text_input('obj', obj)
    model_name = '005/epoch_16.pt'
    model_name = col3.text_input('Clip model_name', model_name)
    if st.button('run_generator'):
        is_run = 1 - is_run
    if is_run == 1:
        out_list = AttDes.validate_local.validate(img1_id, img2_id, obj, model_path)

        sentences1 = out_list[0][0].replace('；', '，').split('，')
        groundtruth1 = ""
        for i in sentences1:
            if obj in i:
                groundtruth1 = i

        sentences2 = out_list[1][0].replace('；', '，').split('，')
        groundtruth2 = ""
        for i in sentences2:
            if obj in i:
                groundtruth2 = i
        col1, col2 = st.columns([1, 1])
        col1.text(out_list[0][0] + '\n' + groundtruth1 + '\n' + out_list[0][1] + '\n' + out_list[0][2] + '\n' + out_list[0][3])
        col2.text(out_list[1][0] + '\n' + groundtruth2 + '\n' + out_list[1][1] + '\n' + out_list[1][2] + '\n' + out_list[1][3])

        texts1 = [groundtruth1, out_list[0][1], out_list[0][2], out_list[0][3]]
        texts2 = [groundtruth2, out_list[1][1], out_list[1][2], out_list[1][3]]


        is_run2 = st.button('run_clip')
        if is_run2 == 1 or len(out_list) > 1:
            probs1, probs2, texts1, texts2 = CLIP.usage.calculate.calculate_similarity(
                model_name, img1_id, img2_id, texts1, texts2
            )

            col3, col4 = st.columns([1, 1])
            col3_txt = ""
            col4_txt = ""
            delta = 0
            G_fb = ""
            for i in range(len(texts1)):
                col3_txt += (str(probs1[i]) + texts1[i]) + '\n'
                if probs1[i][0] - probs1[i][1] > 0.5:
                    if texts1[i] == "":
                        continue
                    if delta < probs1[i][0] - probs1[i][1]:
                        delta = probs1[i][0] - probs1[i][1]
                        G_fb = texts1[i]
            delta = 0
            T_fb = ""
            for i in range(len(texts2)):
                col4_txt += (str(probs2[i]) + texts2[i]) + '\n'
                if probs2[i][1] - probs2[i][0] > 0.5:
                    if texts2[i] == "":
                        continue
                    if delta < probs2[i][1] - probs2[i][0]:
                        delta = probs2[i][1] - probs2[i][0]
                        T_fb = texts2[i]
            col3.text(col3_txt)
            col4.text(col4_txt)

            if len(G_fb) != 0 and len(T_fb) != 0:
                feed_back = "我不要" + G_fb + "，我想要" + T_fb + "。"
            elif len(G_fb) != 0 and len(T_fb) == 0:
                feed_back = "我不要" + G_fb + "。"
            elif len(G_fb) == 0 and len(T_fb) != 0:
                feed_back = "我想要" + T_fb + "。"
            else:
                feed_back = "换一个。"
            st.write(feed_back)




