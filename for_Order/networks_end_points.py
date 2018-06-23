import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import tqdm


def shortest_path(n1, n2):
    #  计算1: 找图中两个点的最短路径
    G = nx.Graph()
    nodes = pd.read_csv("./datasets/lastfm.nodes", sep='\t', header=None)  # lastfm.nodes
    nodes_names = nodes[1].values.tolist()
    G.add_nodes_from(nodes_names)
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']
    for k, v in edges.iterrows():
        G.add_edge(v['a'], v['b'])
    try:
        n = nx.shortest_path_length(G, n1, n2)
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
        spl = dict(nx.single_source_shortest_path_length(G, v))
        spl_sorted = sorted(spl.items(), key=lambda x: x[1], reverse=False)
        save_lengths[v] = spl_sorted[:10]

        for p in spl.values():
            pathlengths.append(p)
    df = pd.DataFrame(save_lengths)
    df.to_csv("./datasets/single_source_shortest_path_length.csv", index=None)
    print("average shortest path length %s" % (sum(pathlengths) / len(pathlengths)))


def statistic(head_num=100):
    # 统计每个点的邻接点的个数
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']
    edges_group = edges.groupby(by='a')
    edges_num = {}
    for x, y in edges_group:
        edges_num[x] = len(y)
    df = pd.DataFrame(edges_num, index=[0]).T.sort_values(by=[0], ascending=False)[:head_num]
    df.to_csv("./datasets/edges_num.csv", header=None)


def draw_networks():
    # 画边图
    G = nx.Graph()
    nodes = pd.read_csv("./datasets/lastfm.nodes", sep='\t', header=None)  # lastfm.nodes
    nodes_sample = nodes.head(1)

    print(len(nodes_sample))
    nodes_names = nodes_sample[1].values.tolist()

    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']
    edges_sample = edges.head(3137)
    print(len(edges_sample))
    edges_list = []

    for k, v in tqdm.tqdm(edges_sample.iterrows()):
        edges_list.append((v['a'], v['b']))

    G.add_nodes_from(nodes_names)
    G.add_edges_from(edges_list)
    nx.connected_components(G)
    plt.figure()
    nx.draw_networkx(G, pos=nx.spring_layout(G))
    plt.savefig("draw_networkx.png")
    plt.close()

    plt.figure()
    nx.draw_networkx_edges(G, pos=nx.spring_layout(G))  # 画出图G
    plt.savefig("draw_networkx_edges.png")
    plt.close()

    plt.figure()
    nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))  # 画出图G
    plt.savefig("draw_networkx_edges.png")
    plt.close()

    # //图或网络中节点的聚类系数。计算公式为：节点u的两个邻居节点间的边数除以((d(u)(d(u)-1)/2)。
    cluster = nx.clustering(G)
    print("图或网络中节点的聚类系数 :\n")
    print(cluster)
    print("- * " * 30)


if __name__ == "__main__":
    method = 'draw_edges_networks'

    if method == 'shortest_path':
        shortest_path(20, 30)

    if method == 'average_shortest_path':
        average_shortest_path()

    if method == 'statistic':
        statistic()

    if method == 'draw_networks':
        draw_networks()
