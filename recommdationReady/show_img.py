import matplotlib.pyplot as plt



def show_img(img_id):
    img_path = '/home/data/zh/fur/processed_img/' + str(img_id) + '.jpg'
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.show()



show_img(550709)

