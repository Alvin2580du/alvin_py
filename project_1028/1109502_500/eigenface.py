from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

m = loadmat("./eigenface/ORL_32x32.mat")
label = m['gnd']
train_X = m['fea']
n = loadmat("./eigenface/train_test_orl.mat")
trainIdx = n['trainIdx']
testIdx = n['testIdx']
trainIdx = [x - 1 for x in trainIdx]
testIdx = [x - 1 for x in testIdx]
X_train = np.reshape(train_X[trainIdx], (train_X[trainIdx].shape[0], train_X[trainIdx].shape[2]))
X_test = np.reshape(train_X[testIdx], (train_X[testIdx].shape[0], train_X[testIdx].shape[2]))
y_train = np.reshape(label[trainIdx], (label[trainIdx].shape[0],))
y_test = np.reshape(label[testIdx], (label[testIdx].shape[0],))
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
target_names = list(([str(i) for i in y_test]))


def plot_gallery(images, titles, file_name, h=32, w=32, rows=3, cols=4):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(cols, rows, i + 1)
        plt.imshow(images[i].reshape((h, w)).T, cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

    plt.savefig("{}".format(file_name))


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)
    true_name = target_names[y_test[i]].rsplit(' ', 1)
    return "p:{},t:{}".format(pred_name[-1], true_name[-1])


n_components = 50
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, 32, 32))
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
clf = MLPClassifier(hidden_layer_sizes=(128,), batch_size=16, early_stopping=True).fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, 'results_1.png', 32, 32)
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, 'results_2.png', 32, 32)
