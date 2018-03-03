import matplotlib.pyplot as plt
import wordcloud


def plot_word_cloud(file_name, savename):
    text = open(file_name, 'r', encoding='utf-8').read()

    wc = wordcloud.WordCloud(background_color="white",
                             max_font_size=20,
                             random_state=1,
                             max_words=100,
                             font_path='pyduyp/preprocessing/msyh.ttf')
    wc.generate(text)
    plt.axis("off")
    plt.figure()
    wc.to_file(savename)
