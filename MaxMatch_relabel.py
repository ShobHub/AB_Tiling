from networkx.algorithms import bipartite
from hopcroftkarp import HopcroftKarp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import pickle 


G = nx.read_gpickle("../ContractedTilling_ET_I=4.gpickle")
Pos = pickle.load(open("../CO_NodesPositions_ET_I=4.txt","rb"))

HC = pickle.load(open("../Hcycle_ET_I=4_d0+d1+d2flips.txt", "rb"))

HC_N=[]
for u,v in HC:
	if(u not in HC_N):
		HC_N.append(u)
	if(v not in HC_N):
		HC_N.append(v)

rm=[]
rm8=[]
for i in G.nodes():
	if(i not in HC_N):
		rm.append(i)

G.remove_nodes_from(rm)
Nodes = list(G.nodes())
l = len(Nodes)

def mapping():
	map = {}
	for j in Nodes:
		flag = True
		while flag == True:
			r = random.randint(0,20000)
			if(r not in map.values()):
				map[j]=r
				flag = False
	return map

for j in range(410,4000):
	print(j)
	map = mapping()
	H = nx.relabel_nodes(G, map)
	
	X, Y = bipartite.sets(H)

	BP = {}
	for i in X:
		BP[i]=0
	for i in Y:
		BP[i]=1
	nx.set_node_attributes(H, BP, name="bipartite")

	GD={}
	for line in bipartite.generate_edgelist(H, data=False):
		l=line.split(' ')
		if(int(l[0]) not in list(GD.keys())):
			GD[int(l[0])] = {int(l[1])}
		else:
			GD[int(l[0])].add(int(l[1]))


	M = HopcroftKarp(GD).maximum_matching(keys_only=True)
	
	rev_map = dict((v,k) for k,v in map.items())

	ME = []
	for u,v in M.items():
		ME.append((rev_map[u],rev_map[v]))
	
	pickle.dump(ME, open('MaxMatchU2_relabel/MaxMatch_%i.txt'%j,'wb'))
	
	'''
	EC=[]
	W=[]
	for u,v in list(G.edges()):
		if((u,v) in ME or (v,u) in ME):
			EC.append('purple')
			W.append(3)
		else:
			EC.append('black')
			W.append(1)
	nx.draw(G, node_size= 0, pos=Pos, edge_color=EC, width=W) 
	plt.show()
	'''
	




