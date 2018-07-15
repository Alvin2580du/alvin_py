import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import random


def shortest_path(n1, n2):
    #  计算1: 找图中两个点的最短路径
    G = nx.Graph()  # 新建一个空的图
    nodes = pd.read_csv("./datasets/lastfm.nodes", sep='\t', header=None)  # lastfm.nodes 读取结点数据
    nodes_names = nodes[1].values.tolist()  # 转为list
    G.add_nodes_from(nodes_names)  # 添加结点数据
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges  读取边信息
    edges.columns = ['a', 'b']  # 修改列名
    for k, v in edges.iterrows():  # 遍历所有行 添加到边里面
        G.add_edge(v['a'], v['b'])
    try:
        n = nx.shortest_path_length(G, n1, n2)  # 计算n1和n2的最短路径
        print(n)
        return n
    except nx.NetworkXNoPath:
        print("No Path .")


def average_shortest_path():
    # 计算2：单源最短路径算法求出节点v到图G每个节点的最短路径
    G = nx.Graph()
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']

    for k, v in edges.iterrows():
        G.add_edge(v['a'], v['b'])

    pathlengths = []
    save_lengths = {}
    for v in G.nodes():
        spl = dict(nx.single_source_shortest_path_length(G, v))  # 单源最短路径算法计算v节点的最短路径
        spl_sorted = sorted(spl.items(), key=lambda x: x[1], reverse=False)  # 对计算的路径做排序，由小到大
        save_lengths[v] = spl_sorted[:10]  # 取最近的10个点

        for p in spl.values():
            pathlengths.append(p)
    df = pd.DataFrame(save_lengths)
    df.to_csv("./datasets/single_source_shortest_path_length.csv", index=None)  # 保存到文件
    print("average shortest path length %s" % (sum(pathlengths) / len(pathlengths)))


def statistic(head_num=100):
    # 统计每个点的邻接点的个数
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']
    edges_group = edges.groupby(by='a')  # 以a列分组
    edges_num = {}
    for x, y in edges_group:
        edges_num[x] = len(y)  # 统计每个结点的粉丝
    df = pd.DataFrame(edges_num, index=[0]).T.sort_values(by=[0], ascending=False)[:head_num]  # 保存前100个到文件中
    df.to_csv("./datasets/edges_num.csv", header=None)


def draw_networks(n=10, m=100):
    # 画边图
    # n 是点的个数，m这个点对应的边的行数
    G = nx.Graph()
    nodes = pd.read_csv("./datasets/lastfm.nodes", sep='\t', header=None)  # lastfm.nodes
    nodes_sample = nodes.sample(n)  # 取前n行数据

    print(len(nodes_sample))
    nodes_names = nodes_sample[1].values.tolist()
    print(nodes_names)
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']
    edges_sample = edges.sample(m)  # 对应的前n行结点数据 对应的边的数据的行数
    print(len(edges_sample))
    edges_list = []

    for k, v in tqdm.tqdm(edges_sample.iterrows()):
        edges_list.append((v['a'], v['b']))

    G.add_nodes_from(nodes_names)  # 添加结点数据
    G.add_edges_from(edges_list)  # 添加边的数据
    nx.connected_components(G)  # 建立点和边的关系
    plt.figure()
    nx.draw_networkx(G, pos=nx.spring_layout(G), nodelist=[''])  # 画出图G
    plt.savefig("draw_networkx_{}.png".format(n))  # 保存到文件
    plt.close()

    plt.figure()
    nx.draw_networkx_edges(G, pos=nx.spring_layout(G))  # 画出图G边图
    plt.savefig("draw_networkx_edges_{}.png".format(n))
    plt.close()

    plt.figure()
    nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))  # 画出图G 点图
    plt.savefig("draw_networkx_nodes_{}.png".format(n))
    plt.close()

    # //图或网络中节点的聚类系数。计算公式为：节点u的两个邻居节点间的边数除以((d(u)(d(u)-1)/2)。
    cluster = nx.clustering(G)
    print("图或网络中节点的聚类系数 :\n")
    print(cluster)
    print("- * " * 30)


def sampler(n=130):
    nodes = pd.read_csv("./datasets/lastfm.nodes", sep='\t', header=None)  # lastfm.nodes
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    fw = open("./datasets/lastfm.edges.sample", 'w', encoding='utf-8')
    fw_node = open("./datasets/lastfm.nodes.sample", 'w', encoding='utf-8')

    nodes_select = nodes.sample(n)
    for row in nodes_select.values:
        fw_node.writelines("{} {}".format(row[0], row[1]) + "\n")
        select_edges = edges[edges[0].isin([row[0]])][1].values.tolist()
        select_edges_sub = random.sample(select_edges, int(len(select_edges) * 0.01))
        for i in select_edges_sub:
            res = "{} {}".format(row[0], i)
            fw.writelines(res + "\n")

            name = nodes[nodes[0].isin([i])][1].values
            nodes_sample = "{} {}".format(i, name[0])
            fw_node.writelines(nodes_sample + "\n")


def draw_networks_sample(n=200):
    # 画边图
    # n 是点的个数，m这个点对应的边的行数
    G = nx.Graph()
    nodes = pd.read_csv("./datasets/lastfm.nodes.sample", sep=' ', header=None)[1].values.tolist()  # lastfm.nodes
    edges_sample = pd.read_csv("./datasets/lastfm.edges.sample", header=None, sep=" ")  # lastfm.edges
    edges_sample.columns = ['a', 'b']
    edges_list = []

    for k, v in tqdm.tqdm(edges_sample.iterrows()):
        edges_list.append((v['a'], v['b']))

    G.add_nodes_from(nodes)  # 添加结点数据
    G.add_edges_from(edges_list)  # 添加边的数据
    nx.connected_components(G)  # 建立点和边的关系

    plt.figure(figsize=(20, 16), dpi=100)
    nx.draw(G,
            pos=nx.spring_layout(G),
            edgelist=None,
            width=0.2,
            edge_color='y',
            style='dashed',
            alpha=1.0,
            edge_cmap=None,
            edge_vmin=None,
            edge_vmax=None,
            ax=None,
            arrows=True,
            label=['MyFun'])  # 画出图G边图
    plt.savefig("sample_draw_networkx_{}.png".format(n))
    print("sample_draw_networkx_{}.png".format(n), "保存成功")
    plt.close()

    plt.figure(figsize=(8, 6), dpi=300)
    nx.draw_networkx_edges(G,
                           pos=nx.spring_layout(G),
                           edgelist=None,
                           width=1.0,
                           edge_color='y',
                           style='dashed',
                           alpha=1.0,
                           edge_cmap=None,
                           edge_vmin=None,
                           edge_vmax=None,
                           ax=None,
                           arrows=True,
                           label=['MyFun'])  # 画出图G边图
    plt.savefig("sample_draw_networkx_edges_{}.png".format(n))
    print("sample_draw_networkx_edges_{}.png 保存成功".format(n))
    plt.close()

    plt.figure(figsize=(20, 16), dpi=100)
    nx.draw_networkx_nodes(G, pos=nx.spring_layout(G),
                           nodelist=None,
                           node_size=50,
                           node_color='r',
                           node_shape='o',
                           alpha=1.0,
                           cmap=None,
                           vmin=None,
                           vmax=None,
                           ax=None,
                           linewidths=None,
                           label=None)  # 画出图G 点图
    plt.savefig("sample_draw_networkx_nodes_{}.png".format(n))
    print("sample_draw_networkx_nodes_{}.png  保存成功".format(n))
    plt.close()

    plt.figure(figsize=(20, 16), dpi=100)
    nx.draw_spring(G, with_labels=True)  # 画出图G 点图
    plt.savefig("draw_spring_{}.png".format(n))
    print(G.nodes)
    print("draw_spring_{}.png  保存成功".format(n))
    plt.close()
    # //图或网络中节点的聚类系数。计算公式为：节点u的两个邻居节点间的边数除以((d(u)(d(u)-1)/2)。
    cluster = nx.clustering(G)
    print("sample_ 图或网络中节点的聚类系数 :\n")
    print(cluster)
    print("- * " * 30)


def ex_1():
    G = nx.grid_2d_graph(5, 5)  # 5x5 grid
    nx.write_adjlist(G, 'aa.txt')
    nx.draw(G, with_labels=True)
    plt.show()


if __name__ == "__main__":

    method = 'draw_networks_sample'
    if method == 'ex_1':
        ex_1()

    # 修改method的值， 运行下面if语句对应的程序
    if method == 'shortest_path':
        n1 = 10
        n2 = 2
        shortest_path(n1, n2)

    if method == 'average_shortest_path':
        average_shortest_path()

    if method == 'statistic':
        statistic()

    if method == 'draw_networks':
        draw_networks()

    if method == 'sampler':
        sampler()

    if method == 'draw_networks_sample':
        draw_networks_sample()
