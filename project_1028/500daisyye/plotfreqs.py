import matplotlib.pyplot as plt
import numpy as np

'''
Plots frequencies
'''


def plotFreqs(frequencies1, frequencies2, frequencies3):
    xPos = np.arange(256)

    plt.figure(figsize=(6, 6))

    # Plot frequency dataset 1
    plt.subplot(3, 1, 1)
    plt.bar(xPos, frequencies1, align='center')

    # Plot frequency dataset 2
    plt.subplot(3, 1, 2)
    if frequencies2 != None:
        plt.bar(xPos, frequencies2, align='center')

    # Plot frequency dataset 3
    plt.subplot(3, 1, 3)
    if frequencies3 != None:
        plt.bar(xPos, frequencies3, align='center')

    plt.show()
