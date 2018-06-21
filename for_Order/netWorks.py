import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def draw_networks():
    G = nx.Graph()
    D = nx.DiGraph()
    M = nx.MultiGraph()

    nodes = pd.read_csv("./datasets/test.data", sep='\t', header=None)  # lastfm.nodes
    nodes_names = nodes[1].values.tolist()
    print(nodes_names[:10])

    edges = pd.read_csv("./datasets/train.data", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']
    edges_list = []
    print(edges.head())

    for k, v in edges.iterrows():
        edges_list.append((v['a'], v['b']))

    print(edges_list[:10])

    G.add_nodes_from(nodes_names)
    G.add_edges_from(edges_list)
    # 访问网络中的结点和边
    node_list = G.nodes()
    edge_list = G.edges()

    nx.connected_components(G)
    nx.draw(G)  # 画出图G
    plt.show()  # 显示出来


def statistic():
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']
    edges_group = edges.groupby(by='a')
    edges_num = {}
    for x, y in edges_group:
        edges_num[x] = len(y)
    df = pd.DataFrame(edges_num, index=[0]).T.sort_values(by=[0],  ascending=False)
    df.to_csv("./datasets/edges_num.csv", header=None)

statistic()





