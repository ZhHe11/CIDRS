import streamlit as st
from PIL import Image
# 设置网页标题
st.title('FEEDBACK_GENERATOR')

st.text('This is some text.')

option = st.selectbox(
    '下拉框',
    ('选项一', '选项二', '选项三'))
st.write('：', option)



add_selectbox = st.sidebar.selectbox(
    "图片来源",
    ("本地上传", "URL")
)
uploaded_file = st.sidebar.file_uploader(label='上传图片')

image = Image.open(r'E:\data\pictures\550567.jpg')
st.image(image, caption='本地图片')

html_temp = """
<div style="background-color:black">
<h2> Streamlit ML APP</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

with st.container():
    tab1, tab2 = st.tabs(["Given", "Target"])
    with tab1:
        if text1 != '':
            filename = r'E:\data\pictures' + '/' + text1 + '.jpg'
            tab1.image(Image.open(filename), caption='img1', use_column_width='always')
    with tab2:
        if text2 != '':
            st.empty()
            filename = r'E:\data\pictures' + '/' + text2 + '.jpg'
            tab2.image(Image.open(filename), caption='img2', use_column_width='always')
