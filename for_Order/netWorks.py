import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import community

help(community)

def ex_1():
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_nodes_from([3, 4, 5, 6])
    G.add_cycle([1, 2, 3, 4])
    G.add_edge(1, 3)
    G.add_edges_from([(3, 5), (3, 6), (6, 7)])
    part = community.best_partition(G)

    mod = community.modularity(part,G)
    print(mod)
ex_1()
exit(1)
def ex_2():
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_nodes_from([3, 4, 5, 6])
    G.add_cycle([1, 2, 3, 4])
    G.add_edge(1, 3)
    G.add_edges_from([(3, 5), (3, 6), (6, 7)])
    G = G.to_undirected()
    nx.draw(G)
    plt.savefig("wuxiangtu.png")
    plt.show()


def ex_3():
    # 计算1：求无向图的任意两点间的最短路径
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (4, 5), (4, 6), (5, 6)])
    path = nx.all_pairs_shortest_path(G)


def ex_4():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edge(1, 2)
    G.add_edge(3, 4)
    try:
        n = nx.shortest_path_length(G, 1, 3)
    except nx.NetworkXNoPath:
        print('No path')


def ex_5():
    import sys

    G = nx.grid_2d_graph(5, 5)  # 5x5 grid
    try:  # Python 2.6+
        nx.write_adjlist(G, sys.stdout)  # write adjacency list to screen
    except TypeError:  # Python 3.x
        nx.write_adjlist(G, sys.stdout.buffer)  # write adjacency list to screen
    # write edgelist to grid.edgelist
    nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
    # read edgelist from grid.edgelist
    H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

    nx.draw(H)
    plt.show()


def ex_6():
    G = nx.lollipop_graph(4, 6)
    pathlengths = []

    for v in G.nodes():
        spl = dict(nx.single_source_shortest_path_length(G, v))
        print('{} {} '.format(v, spl))
        for p in spl:
            pathlengths.append(spl[p])

    print("average shortest path length %s" % (sum(pathlengths) / len(pathlengths)))

    # histogram of path lengths
    dist = {}
    for p in pathlengths:
        if p in dist:
            dist[p] += 1
        else:
            dist[p] = 1

    print('')
    print("length #paths")
    verts = dist.keys()
    for d in sorted(verts):
        print('%s %d' % (d, dist[d]))

    print("radius: %d" % nx.radius(G))
    print("diameter: %d" % nx.diameter(G))
    print("eccentricity: %s" % nx.eccentricity(G))
    print("center: %s" % nx.center(G))
    print("periphery: %s" % nx.periphery(G))
    print("density: %s" % nx.density(G))

    nx.draw(G, with_labels=True)
    plt.show()


def average_shortest_path(limit=10):
    # 计算2：单源最短路径算法求出节点v到图G每个节点的最短路径
    G = nx.Graph()
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']

    for k, v in edges.iterrows():
        G.add_edge(v['a'], v['b'])
        print(v['a'])
        if v['a'] > limit:
            break

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


def shortest_path(n1, n2):
    #  找图中两个点的最短路径
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
        return n
    except nx.NetworkXNoPath:
        print("No Path .")
        return None


def sub_graph():
    G = nx.DiGraph()
    G.add_path([5, 6, 7, 8])
    sub_graph = G.subgraph([5, 6, 8])

    nodes = pd.read_csv("./datasets/lastfm.nodes", sep='\t', header=None)  # lastfm.nodes
    nodes_names = nodes[1].values.tolist()
    edges = pd.read_csv("./datasets/lastfm.edges", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']

    nx.draw(sub_graph)
    plt.savefig("youxiangtu.png")
    plt.show()


def draw_networks(limit=5):
    G = nx.Graph()
    nodes = pd.read_csv("./datasets/test.data", sep='\t', header=None)  # lastfm.nodes
    nodes_names = nodes[1].values.tolist()
    edges = pd.read_csv("./datasets/train.data", header=None, sep=" ")  # lastfm.edges
    edges.columns = ['a', 'b']
    edges_list = []

    for k, v in edges.iterrows():
        if v['a'] > limit:
            break
        edges_list.append((v['a'], v['b']))

    G.add_nodes_from(nodes_names)
    G.add_edges_from(edges_list)
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
    df = pd.DataFrame(edges_num, index=[0]).T.sort_values(by=[0], ascending=False)
    df.to_csv("./datasets/edges_num.csv", header=None)

