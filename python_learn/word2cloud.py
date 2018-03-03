import pandas as pd
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import wordcloud


def plot_word_cloud(file_name, savename):
    text = open(file_name, 'r').read()

    alice_coloring = imread("back.png")

    wc = wordcloud.WordCloud(background_color="white", width=800, height=600,
                             mask=alice_coloring,
                             max_font_size=20,
                             random_state=1,
                             max_words=100,
                             font_path='msyh.ttf')
    wc.generate(text)
    image_colors = wordcloud.ImageColorGenerator(alice_coloring)

    plt.axis("off")
    plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis("off")
    plt.figure(dpi=300)
    plt.axis("off")
    wc.to_file(savename)


