import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
from scipy import sparse
import matplotlib.pylab as plt
import itertools
import numpy as np
import copy
import pickle

HC = pickle.load(open("../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt", "rb"))
with open('../I=4/ET/CO_NodesPositions_ET_I=4.txt', 'rb') as handle:
  pos = pickle.loads(handle.read())

HCN=[]
for u,v in HC:
	if(u not in HCN):
		HCN.append(u)
	if(v not in HCN):
		HCN.append(v)

D = lambda posX, posY: np.linalg.norm(posX-posY)
R=[]
DV={}

c=0
for i in HCN:
	print(c)
	for j in HCN:
		if(i!=j):
			r=np.round(D(pos[i],pos[j]),2)
			flag = False
			for k in DV.keys():
				if(r > k-0.1 and r < k+0.1):         #U1:0.1,  U2:1, SQ64:0.5
					p=k
					flag = True
					break
			if flag:
				DV[p].append((i,j))
			else:
				DV[r] = [(i,j)]
	c=c+1
			
print(sorted(list(DV.keys())))
print(len(DV.keys()))

pickle.dump(DV, open("../I=2/Distance_Vertices.txt", "wb"))
print('*')
Dsfds

gr1=nx.Graph()
gr1.add_edges_from(HC)
nx.draw(gr1,pos,node_size=2, node_color='red',edge_color='purple',width=2) # ,with_labels=True,labels=LA)'''
plt.show()
