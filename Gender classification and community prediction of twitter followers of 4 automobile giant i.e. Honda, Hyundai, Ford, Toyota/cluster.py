"""
cluster.py
"""
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import pickle
import urllib.request
import matplotlib.pyplot as plt

def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter
    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    count = Counter()
    for user in users:
        count.update(user['followers'])
    return count

def can_remove(nodex, nodey, graph):
    nodeEdgeDict = {}
    for node in graph.nodes():
        if (node not in nodeEdgeDict.keys()):
            edgelist = []
        else:
            edgelist = nodeEdgeDict[node]
        edgelist.extend(graph.edges([node]))
        nodeEdgeDict[node]=edgelist   
    if(len(nodeEdgeDict[nodex]) > 1 and len(nodeEdgeDict[nodey]) > 1):
        return True
    else:
        return False

def girvan_newman(G):
    graph_copy = G
    between = nx.edge_betweenness_centrality(graph_copy)
    betweenness = sorted(between.items(),key = lambda x:(-x[1],x[0]))
    for edge in betweenness:
        if(can_remove(edge[0][0],edge[0][1],graph_copy)):
            graph_copy.remove_edge(edge[0][0],edge[0][1])
        if(nx.number_connected_components(graph_copy) > 4):
            break
    components = [c for c in nx.connected_component_subgraphs(graph_copy)]
    return components,graph_copy


def create_graph(users):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)
        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.
    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    import warnings
    warnings.filterwarnings("ignore")
    graph = nx.Graph()
    screen_names = [user['screen_name'] for user in users]
    for i in range(len(screen_names)):
        for key in users[i]['followers']:
            graph.add_edge(screen_names[i],key,color = 'r')
    return graph

def draw_network(graph, users, filename,friendsList):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.
    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    lab={}
    screen_names = [user['screen_name'] for user in users]
    for nodes in graph.nodes():
        if nodes in screen_names:
            lab[nodes] = nodes
        else:
            lab[nodes] = ""
    plt.figure(figsize=(15,15))
    nx.draw_networkx(graph,arrows=False,labels = lab,node_color = 'g',edge_color = 'r',width = 0.5,node_size = 20)
    plt.axis('off') 
    plt.savefig(filename)

def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter
    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    count = Counter()
    for i in range(len(users)):
        count.update(users[i]['followers'])
    return count

def main():
    Users = open('users.pkl','rb')
    users = pickle.load(Users)
    friendList = pickle.load(open('friendlist.pkl','rb'))
    friend_counts = count_friends(users)
    graph = create_graph(users)
    draw_network(graph,users,'network.png',friendList)
    clusters,graph_copy = girvan_newman(graph)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes %d %d' %
          (clusters[0].order(), clusters[1].order(),clusters[2].order(),clusters[3].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())
    draw_network(graph_copy,users,'networkoutput.png',friendList)
    with open('graph.pkl', 'wb') as fp:
        pickle.dump(graph_copy, fp)
if __name__ == '__main__':
    main()