import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
from scipy import sparse
from scipy.spatial import KDTree as KD
from operator import itemgetter
import geometry_sp as geo
import graph_mincopy as GC
import matplotlib.pylab as plt
import numpy as np
import pickle
import time
import os, psutil

def del_all(mapping, to_remove):
      for key in to_remove:
          del mapping[key]

st = time.time()

T = geo.Tiling("eightemp", inflations=3)    #"vertexconfig",8
process = psutil.Process(); print(process.memory_info().rss)
Adj_Decorate,Adj_Full,Pos_Decorate, BN,P=T.decorate()

process = psutil.Process(); print(process.memory_info().rss)

CNV = {}
while bool(BN) == True:
	l=[]
	fk = next(iter(BN))
	qv = BN[fk]
	del BN[fk]
	
	t = KD(list(BN.values()))
	a, b = t.query([qv], distance_upper_bound=0.04, k=15)

	for i in set(b[0]):
		if( i < len(list(BN.keys())) ):
			c = list(BN.keys())[i]
			l.append(c)
			CNV[c] = fk
	del_all(BN, l)

#print(CNV) 
	
edgesF = []
for u,v in Adj_Full:
	if( u in CNV.keys()):
		u = CNV[u]
	if( v in CNV.keys()):
		v = CNV[v]
	edgesF.append((u,v))

edgesD = []
for u,v in Adj_Decorate:
	if( u in CNV.keys()):
		u = CNV[u]
	if( v in CNV.keys()):
		v = CNV[v]
	edgesD.append((u,v))
g = nx.Graph()
g.add_edges_from(edgesF)
print(len(g.nodes()))

print(time.time() - st)

EC=[]
W=[]
for u,v in g.edges():
	if((u,v) in edgesD or (v,u) in edgesD):
		EC.append('purple')
		W.append(2)
	else:
		EC.append('black')
		W.append(1)
nx.draw(g, pos=Pos_Decorate, node_size=0, edge_color=EC, width = W) #, with_labels=True, font_size = 10)
plt.show()


#nx.write_gpickle(gr,'../AB_HNew/ContractedTilling_ET_I=4.gpickle')
#with open('../AB_HNew/CO_NodesPositions_ET_I=4.txt', 'wb') as handle:
#  pickle.dump(Pos_Decorate, handle)

