import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
from scipy import sparse
import geometry_sp as geo
import graph_mincopy as GC
import matplotlib.pylab as plt
import numpy as np
import pickle
import time
import os, psutil

## Here inflations = 3 means 8 times inflated (files are wrongly marked - so we get file with 'VC_I=4' by putting inflation = 2 here, which means I should 6 not 4 - remember for future!!)

def ContractNodes(a,L,G):
	e=[]
	for i in L:
		e.extend(list(G.edges(i)))
	G.remove_nodes_from(L)
	for i in e:
		if(i[0]in L):
			G.add_edge(a,i[1])
		elif(i[1] in L):
			G.add_edge(a,i[0])

st = time.time()

T = geo.Tiling("eightemp", inflations=2)    #"vertexconfig",8
process = psutil.Process(); print(process.memory_info().rss)
Adj_Decorate,Adj_Full,Pos_Decorate, bound_nodes,P=T.decorate()

process = psutil.Process(); print(process.memory_info().rss)

gr=nx.Graph()
#rows, cols = np.where(Adj_Full == 1)
#edgesF = zip(rows.tolist(), cols.tolist())
gr.add_edges_from(Adj_Full)   #edgesF)

#nx.draw(gr, pos=Pos_Decorate, node_size=0)
#plt.show()

#dsfdsf

CN={}
CNV=[]
for i in bound_nodes.keys():
	if(i not in CNV):
		CNL=[]
		CNV.append(i)
		for j in set(bound_nodes.keys())-set(CNV):
			if(np.linalg.norm(bound_nodes[i]-bound_nodes[j])<=0.04):
				CNL.append(j)
				CNV.append(j)
		CN[i]=CNL

inv_CN = {i: k for k, v in CN.items() for i in v}

for i in CN.keys():
	if(len(CN[i])!=0):
		ContractNodes(i,CN[i],gr)

print(len(gr.nodes()))

#nx.write_gpickle(gr,'../AB_HNew/ContractedTilling_ET_I=4.gpickle')
#with open('../AB_HNew/CO_NodesPositions_ET_I=4.txt', 'wb') as handle:
#  pickle.dump(Pos_Decorate, handle)

#rowsD, colsD = np.where(Adj_Decorate == 1)
edgesD = Adj_Decorate           #zip(rowsD.tolist(), colsD.tolist())

edgesDC=[]
for u,v in edgesD:
	if( u in inv_CN.keys()):
		u = inv_CN[u]
	if( v in inv_CN.keys()):
		v = inv_CN[v]
	edgesDC.append((u,v))

#pickle.dump(edgesDC, open("../AB_HNew/DecorateEdge_ET_I=4.txt", "wb"))

process = psutil.Process(); print(process.memory_info().rss)
print(time.time()-st)

EC=[]
W=[]
for u,v in gr.edges():
	if((u,v) in edgesDC):
		EC.append('purple')
		W.append(2)
	elif((v,u) in edgesDC):
		EC.append('purple')
		W.append(2)
	else:
		EC.append('black')
		W.append(1)



nx.draw(gr, pos=Pos_Decorate, node_size=0, edge_color=EC, width=W) #,with_labels=True,labels=LA)
plt.show()

#

'''
for i in P:
	for j in i:
		if(j in CNV):
			i[i.index(j)]=inv_CN[j]

#print(P)		
pickle.dump(P, open("../AB_HNew/PathPoints_ET_I=4.txt", "wb"))
'''

'''
L=len(Adj_Decorate)
for i in CN.keys():
	for j in CN[i]:
		A=np.where(Adj_Decorate[j]==1)  #[i if B[k]==1 for k in range(L)]
		Adj_Decorate[i][A.tolist()]=1

		A=np.where(Adj_Decorate[:,j]==1)  #[i if B[k]==1 for k in range(L)]
		Adj_Decorate[:,i][A.tolist()]=1
'''
