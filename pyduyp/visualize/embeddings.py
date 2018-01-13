import pandas as pd
import matplotlib.pyplot as plt


# visualize the file info
def __visualize(path):
    df = pd.read_csv(path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(df['x'].values, df['y'].values, maker='o', alpha=0.0)
    for k, v in df.iterrows():
        ax.annotate(v['word'], [v['x'], v['y']])
    plt.title('Embedding words')
    plt.grid()
    plt.show()
