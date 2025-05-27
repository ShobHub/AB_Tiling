import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
from scipy import sparse
import geometry_sp as geo
import graph_mincopy as GC
import matplotlib.pylab as plt
import numpy as np
import pickle
import time

## Here inflations = 3 means 8 times inflated (files are wrongly marked - so we get file with 'VC_I=4' by putting inflation = 2 here, which means I should 6 not 4 - remember for future!!)

st = time.time()

T = geo.Tiling("eightemp",inflations=1)      #vertexconfig",8,inflations=1)
Adj_Decorate,Adj_Full,Pos_Decorate, bound_nodes,P=T.decorate()

#LA={(i):(i if i<=5500 else '') for i in range(len(Adj_Full))}

gr=nx.Graph()
rows, cols = np.where(Adj_Full == 1)
edgesF = zip(rows.tolist(), cols.tolist())
gr.add_edges_from(edgesF)

#nx.draw(gr,Pos_Decorate,node_size=2, node_color='red') #,with_labels=True,labels=LA)
#plt.show()

st = time.time()

CN={}
CNV=[]
for i in bound_nodes.keys():
	if(i not in CNV):
		CNL=[]
		for j in bound_nodes.keys():
			if(np.linalg.norm(bound_nodes[i]-bound_nodes[j])<=0.04 and i!=j):
				CNL.append(j)
				CNV.append(j)
		CN[i]=CNL

CN={k: v for k, v in CN.items() if v}
inv_CN = {i: k for k, v in CN.items() for i in v}

for i in CN.keys():
	if(len(CN[i])==1):
		#print(i,CN[i][0])
		gr = nx.contracted_nodes(gr, i, CN[i][0])
		#del LA[CN[i][0]]
		del Pos_Decorate[CN[i][0]]
	else:
		for node in CN[i]:
			#print(i, node)
			gr = nx.contracted_nodes(gr, i, node)
			#del LA[node]
			del Pos_Decorate[node]

#print(Pos_Decorate)

nx.draw(gr,Pos_Decorate,node_size=2, node_color='red') #,with_labels=True,labels=LA)  #,edge_color=Emap, width=W) 
plt.show()
print(len(gr.nodes()))

#nx.write_gpickle(gr,'../AB_HNew/ContractedTilling_ET_I=2.gpickle')
#with open('../AB_HNew/CO_NodesPositions_ET_I=2.txt', 'wb') as handle:
#  pickle.dump(Pos_Decorate, handle)

rowsD, colsD = np.where(Adj_Decorate == 1)
edgesD = zip(rowsD.tolist(), colsD.tolist())

edgesDC=[]
for i in edgesD:
	c=0
	if(i[0] in CNV):
		c=c+1
	if(i[1] in CNV):
		c=c+2
	if(c==0):
		edgesDC.append(i)
	elif(c==1):
		edgesDC.append((inv_CN[i[0]],i[1]))
	elif(c==2):
		edgesDC.append((i[0],inv_CN[i[1]]))
	elif(c==3):
		edgesDC.append((inv_CN[i[0]],inv_CN[i[1]]))

#print(edgesDC)
	
#pickle.dump(edgesDC, open("../AB_HNew/DecorateEdge_ET_I=2.txt", "wb"))
print(time.time()-st)

'''
for i in P:
	for j in i:
		if(j in CNV):
			i[i.index(j)]=inv_CN[j]

#print(P)		
pickle.dump(P, open("../AB_HNew/PathPoints_ET_I=8.txt", "wb"))

plt.show()
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
