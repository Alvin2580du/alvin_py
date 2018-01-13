import pandas as pd
import os
from scipy.misc import imread
import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator


def plot_word_cloud():
    data = pd.read_csv("./datasets/question_45/cd_cut.csv", usecols=['msg_cut'])
    data.to_csv("./datasets/question_45/cd_cut_new.csv", index=None)
    text = open("./datasets/question_45/cd_cut_new.csv", 'r').readlines()

    alice_coloring = imread(os.path.join("./datasets/question_45/results", "alice_color.png"))

    wc = WordCloud(background_color="white",  # 背景颜色max_words=2000,# 词云显示的最大词数
                   mask=alice_coloring,  # 设置背景图片
                   max_font_size=40,  # 字体最大值
                   random_state=42)
    wc.generate(text)
    image_colors = ImageColorGenerator(alice_coloring)

    plt.axis("off")
    plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis("off")
    plt.figure()
    plt.axis("off")
    wc.to_file(os.path.join("./datasets/question_45/results", "traffic.png"))
