import pandas as pd
import os


def add_(inputs):
    return inputs - 1


# Length	Height	Width

for file in os.listdir("."):
    if '.xls' in file:
        data = pd.read_excel("{}".format(file))
        data['Length'] = data['Length'].apply(add_)
        data['Height'] = data['Height'].apply(add_)
        data['Width'] = data['Width'].apply(add_)
        # data['Received Status'] = ['Y'] * data.shape[0]
        data.to_excel("./OK/{}".format(file), index=None)
        print("./OK/{}".format(file))

