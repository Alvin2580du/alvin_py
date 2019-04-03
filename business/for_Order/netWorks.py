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


def draw_edges_networks():
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
    nx.draw_networkx(G, pos=nx.spring_layout(G), nodelist=[])
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

    # 计算图的密度，其值为边数m除以图中可能边数（即n(n-1)/2）
    # d = nx.degree(G)
    # print("图的密度:\n")
    # print(d)
    # print("- * " * 30)
    # 节点度中心系数。通过节点的度表示节点在图中的重要性，默认情况下会进行归一化，
    # 其值表达为节点度d(u)除以n-1（其中n-1就是归一化使用的常量）。这里由于可能存在循环，所以该值可能大于1.
    # dc = nx.degree_centrality(G)
    # print("节点度中心系数:\n")
    # print(dc)
    # 节点介数中心系数。在无向图中，该值表示为节点作占最短路径的个数除以((n-1)(n-2)/2)；在有向图中，该值表达为节点作占最短路径个数除以((n-1)(n-2))。
    # bc = nx.betweenness_centrality(G)
    # print("节点介数中心系数 :\n")
    # print(bc)
    # print("- * " * 30)

    # //图或网络的传递性。即图或网络中，认识同一个节点的两个节点也可能认识双方，计算公式为3*图中三角形的个数/三元组个数（该三元组个数是有公共顶点的边对数，这样就好数了）。
    # tran = nx.transitivity(G)
    # print("图或网络的传递性 :\n")
    # print(tran)
    # print("- * " * 30)

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
        draw_edges_networks()
