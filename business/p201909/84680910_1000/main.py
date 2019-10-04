# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import time

from math import cos, sin, atan2, sqrt, pi, radians, degrees


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance / 111


def build_kDistance(path):
    """
    综合图，可以选择得到半径Eps的范围大致在0.002~0.006之间：
    """

    df = pd.read_csv(path)
    num = 0
    k_dis = []
    start = time.time()
    for x, y in df.iterrows():
        num += 1
        ds_ = 1000000000000
        for x2, y2 in df.iterrows():
            lng1 = y['longitude']
            lat1 = y['latitude']
            lng2 = y2['longitude']
            lat2 = y2['latitude']
            if lng1 == lng2 and lat1 == lat2:
                continue
            ds = geodistance(lng1, lat1, lng2, lat2)

            if ds < ds_:
                ds_ = ds

        k_dis.append(ds_)

        if num % 50 == 1:
            print('{}/{}'.format(num, df.shape[0]), "{:0.3f}".format(time.time() - start))

    df = pd.DataFrame({"k_dis": k_dis})
    df.to_csv("k_dis_{}.csv".format(path.replace('.csv', '')), index=None)

    plt.figure(dpi=200, figsize=(15, 8))
    plt.scatter(range(len(k_dis)), sorted(k_dis))
    plt.title("K-distance scatter plot- {}".format(path.replace('.csv', '')))
    plt.savefig("K-distance scatter plot - {}.png".format(path.replace('.csv', '')))
    print("K 距离图保存成功！")


def center_geolocation(geolocations):
    x = 0
    y = 0
    z = 0
    lenth = len(geolocations)
    for lon, lat in geolocations:
        lon = radians(float(lon))
        lat = radians(float(lat))
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)

    x = float(x / lenth)
    y = float(y / lenth)
    z = float(z / lenth)
    return (degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y))))


def build(path):
    eps = 0.0035
    min_samples = 200
    df = pd.read_csv(path, sep=',')
    X = df.as_matrix(columns=['longitude', 'latitude'])

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean',
                metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)

    n_noise_ = list(labels).count(-1)
    print("n_noise_", n_noise_)
    # Plot result
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    print("unique_labels", unique_labels)
    colors = [each for each in np.linspace(0, 1, len(unique_labels))]
    print(colors)
    colors = [plt.cm.Spectral(each) for each in colors]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)

        xy = X[class_member_mask & ~core_samples_mask]

        df2 = pd.DataFrame(xy)
        df2.to_csv('{}_{}.csv'.format(k, path.replace('.csv', '')), index=None)
        xy_centers = center_geolocation(df2.values.tolist())
        print("{} 类:{} 数量：{}， 中心点: {} ".format(path.replace('.csv', ''), k, df2.shape[0], xy_centers))

        plt.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor="#808080", markeredgecolor='#808080', markersize=0.3)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    # xticks = [103.8, 103.9, 104.0, 104.1, 104.2, 104.3, 104.4]
    # plt.xticks(xticks, rotation=0)
    plt.savefig("{}_results.png".format(path.replace('.csv', '')))


if __name__ == '__main__':

    method = 'build_kDistance'

    if method == 'txt2csv':
        path = '早高峰.txt'
        save_name = 'zaoGF.csv'  # wanGF.csv, yeGF.csv, zaoGF.csv

        with open(path, 'r', encoding='gbk') as fr:
            lines = fr.readlines()
            save = []
            for line in lines:
                x, y = line.split(',')[0], line.split(',')[1]
                rows = {'longitude': x.replace('\n', ''), 'latitude': y.replace('\n', '')}
                save.append(rows)
            df = pd.DataFrame(save)
            df.to_csv(save_name, index=None)

    if method == 'build_kDistance':
        path = "yeGF.csv"  # wanGF.csv, yeGF.csv, zaoGF.csv
        build_kDistance(path)

    if method == 'build':
        path = "yeGF.csv"  # wanGF.csv, yeGF.csv, zaoGF.csv
        build(path)
