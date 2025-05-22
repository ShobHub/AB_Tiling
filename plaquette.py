import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
from scipy import sparse
#import geometry_sp as geo
#import graph_mincopy as GC
import matplotlib.pylab as plt
import itertools
import numpy as np
import copy
import pickle

# gr = pickle.load(open('../I=6/ContractedTilling_ET_I=6.gpickle','rb'))
# with open('../I=6/CO_NodesPositions_ET_I=6.txt', 'rb') as handle:
#   pos = pickle.loads(handle.read())

# gr = pickle.load(open('../../Penrose/Cen_Tiling_(k,R)=(20,20).txt','rb'))
# pos = pickle.load(open('../../Penrose/Cen_Pos_(k,R)=(20,20).txt','rb'))

# gr = pickle.load(open('../../HatTile/Hat/Hout2.txt','rb'))
# pos = pickle.load(open('../../HatTile/Hat/Pos_Hout2.txt','rb'))

# gr = pickle.load(open('../../Penrose/ModifiedTiling/Cen_Mod_Tiling_(k,R)=(20,20).txt','rb'))
# pos = pickle.load(open('../../Penrose/ModifiedTiling/Cen_Pos_Mod_Tiling_(k,R)=(20,20).txt','rb'))

# gr = pickle.load(open('../../Penrose/3DegLattices/SqOct/SqOct12_PBC.txt','rb'))
# pos = pickle.load(open('../../Penrose/3DegLattices/SqOct/Pos_SqOct12_PBC.txt','rb'))

# gr = pickle.load(open('../../Penrose/3DegLattices/Hexa/Hexa30_PBC.txt','rb'))
# pos = pickle.load(open('../../Penrose/3DegLattices/Hexa/Pos_Hexa30_PBC.txt','rb'))

gr = pickle.load(open('../../Penrose/3DegLattices/(4,6)_OBC.txt','rb'))
pos = pickle.load(open('../../Penrose/3DegLattices/Pos_(4,6)_OBC.txt','rb'))

def findPaths(G,u,n):
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1)]
    return paths
def find_cycles(G,u,n):
    paths = findPaths(gr,u,n)
    return [tuple(path) for path in paths if (path[-1] == u) and sum(x ==u for x in path) == 2]

Le = [4]  #,8]
P = []
P_sort = []
print(len(gr.nodes()))
for k in Le:
    print(k)
    for i in list(gr.nodes()):
        cycles = find_cycles(gr,i,k)  
        #print(i)
        #print(len(cycles))
        for j in cycles:
            if(len(set(j))==k and tuple(set(sorted(j))) not in P_sort):     #4,14
                P.append(j)
                P_sort.append(tuple(set(sorted(j))))
                #a,b,c,d,e=j
                #P.append(a)
                #P.append(b)
                #P.append(c)
                #P.append(d)
                #P.append(e)

PL_edge=[]
for i in P:
	E=[]
	for j in range(len(i)-1):
		E.append((i[j],i[j+1]))
	PL_edge.append(E)

print(PL_edge)
print(P)
print(len(PL_edge),len(P))
pickle.dump(PL_edge, open("../../Penrose/3DegLattices/Plaqs_(4,6)_OBC.txt", "wb"))  #26808 ET(I=4)
gr1=nx.Graph()
for i in PL_edge:
	gr1.add_edges_from(i)
print(gr==gr1,nx.is_isomorphic(gr,gr1))
nx.draw(gr1, pos, node_size=5)   #, with_labels=True)
plt.show()
