import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd


def draw(plt, values, type, line_style, color, legend):
    plt.plot(np.arange(len(values)), values, type, linestyle=line_style, color=color, label=legend)


if __name__ == '__main__':
    file_names = ["/home/dms/mxnet_face/vgg16_100_epoch_result/vgg_16_reduced.log",
                  "/home/dms/mxnet_face/inceptionbn_100_epoch_result/inception_bn.log"]
    # file_names = ["/home/dms/mxnet_face/vgg16_100_epoch_result/vgg_16_reduced_fine.log",
    #               "/home/dms/mxnet_face/inceptionbn_100_epoch_result/inception_bn_fine.log"]

    types = ['-', '-']
    plt.figure(figsize=(14, 10))

    plt.xlabel("Epoch")
    plt.ylabel("RMSE")

    for i, file_name in enumerate(file_names):

        log = open(file_name).read()
# 2017-04-01 15:20:08,108 Node[0] Epoch[0] Batch [2]	Speed: 130.14 samples/sec	Train-nrmse=1.739190
# 2017-04-01 15:20:08,109 Node[0] Epoch[0] Batch [2]	Speed: 130.14 samples/sec	Train-rmse=0.506505
        log_tr = re.compile('.*Epoch\[(\d+)\].*Batch \[(\d+)\].*Train-rmse=([-+]?\d*\.\d+|\d+)').findall(log)

        log_va = re.compile('.*Epoch\[(\d+)\].*Validation-rmse=([-+]?\d*\.\d+|\d+)').findall(log)

        log_n_tr = re.compile('.*Epoch\[(\d+)\].*Batch \[(\d+)\].*Train-nrmse=([-+]?\d*\.\d+|\d+)').findall(log)

        log_n_va = re.compile('.*Epoch\[(\d+)\].*Validation-nrmse=([-+]?\d*\.\d+|\d+)').findall(log)

        log_tr = np.array(log_tr)

        log_n_tr = np.array(log_n_tr)

        for n in range(i):
            log_tr_ = pd.DataFrame(log_tr)
            log_n_tr_ = pd.DataFrame(log_n_tr)
            log_tr_.to_csv("../results/1 log_tr.csv", sep="\t")
            log_n_tr_.to_csv("../results/2 log_n_tr.csv", sep="\t")

        data = {}
        for epoch, batch, rmse in log_tr:
            if len(data) == 0 or int(epoch) is not data[len(data) - 1][0]:
                data[len(data)] = [int(epoch), float(rmse), 1]
            else:
                data[len(data) - 1][1] += float(rmse)
                data[len(data) - 1][2] += 1

        tr_value = []
        for vals in data:
            tr_value.append(data[vals][1] / data[vals][2])
        pd.DataFrame(tr_value).to_csv("../results/1.1tr_value.csv", sep="\t")

        data = {}
        for epoch, batch, rmse in log_n_tr:
            if len(data) == 0 or int(epoch) is not data[len(data) - 1][0]:
                data[len(data)] = [int(epoch), float(rmse), 1]
            else:
                data[len(data) - 1][1] += float(rmse)
                data[len(data) - 1][2] += 1
        # 2
        n_tr_value = []
        for vals in data:
            n_tr_value.append(data[vals][1] / data[vals][2])
        pd.DataFrame(n_tr_value).to_csv("../results/2.1 n_tr_value.csv", sep="\t")

        # 3
        va_value = []
        for vals in log_va:
            va_value.append(vals[1])
        pd.DataFrame(va_value).to_csv("../results/1.2 va_value.csv", sep="\t")
        # 4
        n_va_value = []
        for vals in log_n_va:
            n_va_value.append(vals[1])
        pd.DataFrame(n_va_value).to_csv("../results/2.2 n_va_value.csv", sep="\t")

        legend_name = file_names[int("%d" % i)].split("/")[-1]

        draw(plt, tr_value, types[i], '-', 'r', "Train-RMSE/image size- " + legend_name)

        draw(plt, va_value, types[i], '-', 'b', "Validation-RMSE/image size- " + legend_name)

        draw(plt, n_tr_value, types[i], '--', 'r', "Train-RMSE/iod- " + legend_name)

        draw(plt, n_va_value, types[i], '--', 'b', "Validation-RMSE/iod- " + legend_name)

    plt.legend(loc="best")

    plt.yticks(np.arange(0, 0.2, 0.05))
    plt.ylim([0, 0.2])
    plt.savefig("../results/loss1.png", dpi=600)

