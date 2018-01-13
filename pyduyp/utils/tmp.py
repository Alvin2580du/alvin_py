import matplotlib.pyplot as plt
import numpy as np


def ubiform_plot():
    # 均匀分布
    s = np.random.uniform(-1, 0, 1000)
    count, bins, ignored = plt.hist(s, 15, normed=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()
