'''
jerome.lloyd@unige.ch
date created: 2020.09.10

------------------------
graph from geometry, using networkX package.
------------------------
'''

import sys
import copy
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import geometry_sp as geo  ## this package builds the adjacency matrix for AB

np.set_printoptions(threshold=sys.maxsize, linewidth=1e5)
pi = np.pi
s_mean = 1+np.sqrt(2)

class TilingGraph():

    def __init__(self, T, bipartiteedges=True):

        if isinstance(T, geo.Tiling):
            self.inflations = T.inflations
            self.charge = T.charge
            
            if T.inflations == 0:
                T.to_sparse()
            if bipartiteedges:
                T.remove_non_bipartite()
            self.adj = T.adj.copy()
            self.pos = T.pos.copy()

            ''' build networkx graph from T adjacency matrix'''
            self.G = nx.OrderedGraph()
            for idx, node_list in enumerate(self.adj):
                edge_list = [(idx, node) for node in node_list]
                self.G.add_edges_from(edge_list)

            if bipartiteedges:
                nx.set_node_attributes(self.G, bipartite.color(self.G), name='charge')
            nx.set_node_attributes(self.G, dict(enumerate(self.pos)), name='pos')  # node positions 
            nx.set_node_attributes(self.G, T.eightsgen, name='eightsgen')  # generation of eight vertices (might be broken..)

        elif isinstance(T, nx.Graph):
            self.G = T.copy()


    def maxmatch(self, doprune=False, init=False, largest=True):
        '''
        find maximum matching of G.
        doprune: bool: if True, prune twigs (uneccessary for square and eight-empire graphs)
        '''

        if doprune is True:
            self.prune()
        if init is True:
            self.G = nx.convert_node_labels_to_integers(self.G)

        G = self.G
        self.monomers = list()
        self.dimers = dict()
        nx.set_node_attributes(G, bipartite.color(G), name='bipartite')
        nx.set_node_attributes(G, 0, name='connected')
        nx.set_edge_attributes(G, 0, name='dimer')

        graphs = nx.connected_components(G)

        if largest is True:
            largest_cc = max(nx.connected_components(G), key=len)
            graph_cc = [largest_cc]
        else:
            graph_cc = graphs

        for cc in graph_cc:
            g = G.subgraph(cc).copy()
            dimers = nx.bipartite.hopcroft_karp_matching(g)

            for key, value in dimers.items():
                G.nodes[key]['connected'] = 1
                G.nodes[value]['connected'] = 1
                G[key][value]['dimer'] = 1

            monomers = [x for x, y in G.nodes(data=True) if y['connected'] == 0]

            self.G = G
        self.monomers += monomers
        self.dimers = {**self.dimers, **dimers}

        self.isperfectmatch = nx.is_perfect_matching(G, dimers)
        #self.G, self.monomers, self.dimers = G, monomers, dimers

        return G, monomers, dimers

    def prune(self):
        '''
        remove twigs. pruning intermediate inflations will result in errors
        '''
        leave, H = False, nx.Graph()
        while leave is False:
            leave, H = True, copy.deepcopy(self.G)
            for node in H:
                if H.degree(node) == 1:
                    self.G.remove_node(node)
                    leave = False
        if nx.is_connected(self.G)is False:
            Gc = max(nx.connected_component_subgraphs(self.G), key=len)
            self.G.remove_nodes_from(list(set(self.G)-set(Gc)))

        return



    def draw_graph(self, draw_matching=False, color_nodes=False, savename=None, draw=True):

        G = self.G

        nodes = G.nodes()
        edges = G.edges()
        nnodes = len(nodes)
        pos = nx.get_node_attributes(G, 'pos')  # extract data from graph

        scale = nnodes/50  # scale resolution according to number of nodes
        edge_width = 1/np.sqrt(scale)
        node_base_size = 20/scale**2
        node_base_color = 'k'
        edge_base_color = 'k'
        edge_base_width = .3
        node_color = [node_base_color for n in nodes]
        node_size = [node_base_size for n in nodes]
        edge_color = [edge_base_color for e in edges]
        edge_width = [edge_base_width for e in edges]

        if color_nodes:  # colour nodes by bipartite charge
            charge = nx.get_node_attributes(G, 'charge')
            node_color = ['r' if charge[n] == 0 else 'b' for n in nodes]

        if draw_matching:
            G, monomers, dimers = self.maxmatch()
            monomer_charge = [G.nodes[m]['bipartite'] for m in monomers]
            red_monomers = monomer_charge.count(1)
            blue_monomers = monomer_charge.count(0)
            dimers = [G[u][v]['dimer'] for u, v in edges]
            edge_color = ['m' if d == 1 else edge_base_color for d in dimers]
            edge_width = [5.*edge_base_width if d == 1 else edge_base_width for d in dimers]

            for idx, m in enumerate(monomers):
                charge_color = {0:'b', 1:'r'}
                node_size[list(nodes).index(m)] = node_base_size*5
                node_color[list(nodes).index(m)] = charge_color[monomer_charge[idx]]


        fs = 5  # figsize in inches
        fig, ax = plt.subplots(1, figsize=(fs, fs), dpi=200)
        ax.axis = 'equal'

        kwargs = {'ax':ax, 'node_size':node_size, 'width':edge_width, 'node_color':node_color, 'edge_color':edge_color,
                  'linewidths':1}
       
        nx.draw(G, pos,with_labels=True, labels=T.labels,**kwargs)

        if savename is not None:
            plt.savefig('{}.pdf'.format(savename), dpi='figure')
        if draw is True:
            plt.show()

        return




if __name__ == "__main__":

    '''
    the geometry_sp module does all the hard work, building the adjacency matrix. Let's generate a single square tile, then inflate it three times using the inflation rules:
    '''
    #T = geo.Tiling("square")
    #T.inflate(3)
    '''
    T contains the graph geometry in a single adjacency matrix: for a graph with N nodes, it is a NxN matrix, but in practice it is stored as a sparse
    matrix (so we store a list of neighbours for each node). After generating, we can convert to a networkx object (this gives many useful tools
    and allows e.g. easy plotting, but it is much faster to do inflation on the sparse matrix rather than on a networkx graph). Let's plot the
    graph:
    '''

    #t = TilingGraph(T)
    #t.draw_graph()

    '''
    you can use a few different seeds to start the inflation. for example, the 'eight-empire':
    '''

    T = geo.Tiling("eightemp")
    t = TilingGraph(T)
    t.draw_graph()

    '''
    or one of the seven vertex configuration:
    '''
    #T = geo.Tiling("vertexconfig", 6, inflations=4)
    #t = TilingGraph(T)
    #t.draw_graph()

    '''
    finally let's illustrate a matching configuration for the square(3). We use the standard Hopcroft-Karp algorithm (time ~ poly(#vertices)) to generate a
    set of dimers and monomers (if they exist). The square(3) admits a perfect matching, but square(4) does not... :
    '''
    #T = geo.Tiling("square", inflations=3)
    #t = TilingGraph(T)
    #t.draw_graph(draw_matching=True)


    '''
    there is much more you can do... membranes, conformal mappings etc. This should be enough to get started with the tiling.
    '''

    plt.show()
