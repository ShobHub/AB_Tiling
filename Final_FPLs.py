import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
from scipy import sparse
import geometry_sp as geo
import graph_mincopy as GC
import matplotlib.pylab as plt
import itertools
import numpy as np
import copy
import pickle

#FPLs = pickle.load(open("FPLsConfig_VC_I=4.txt", "rb"))
PL = pickle.load(open("PlaquetteCycles_VC_I=4.txt", "rb"))
HC = pickle.load(open("HCycle_VC_I=4.txt", "rb"))
HC1=HC

#print(len(FPLs))
#HC=FPLs[299]

with open('CO_NodesPositions_VC_I=4.txt', 'rb') as handle:
  pos = pickle.loads(handle.read())

D = lambda posX, posY: np.linalg.norm(posX-posY)
HCN=[]
for i in HC:
	HCN.append(i[0])
	HCN.append(i[1])

PLH=[]
for i in PL:
	c=0
	for j in i:
		if(j[0] in HCN and j[1] in HCN):
			c=c+1
	if(c==4):
		PLH.append(i)

def P_inHC(i,HC):
	c=0
	for u,v in i:
		if((u,v) in HC or (v,u) in HC):
			c=c+1
	return c
	

def flip_plaq(j,HC):
	u,v=j
	for i in PLH:
		c=P_inHC(i,HC)
		if(c==2):
			if ((u,v) in i or (v,u) in i):
				e=[k for k in i if((k[0]!=u and k[1]!=v and (k[0],u) not in HC) and (k[1]!=u and k[0]!=v))]
				if(e!=[]):
					u1,v1=e[0]
					if((u1,v1) in HC):
						#print(i,j)
						return (u1,v1)
					elif((v1,u1) in HC):
						#print(i,j)
						return (v1,u1)

FPLs=[]
#FPLs.append(HC1)
#HC1=FPLs[1999]


nflips=1
while(nflips<2):
	#print(nflips)
	#n=np.random.randint(1, len(HC))
	#se=HC[n]
	for i in HC1:
		print(nflips)
		se=i
		e=flip_plaq(se,HC)
		if(e!=None):
			nflips=nflips+1
			HC.remove(se)
			HC.remove(e)
			if(D(pos[se[0]],pos[e[0]])<D(pos[se[0]],pos[e[1]]) and D(pos[se[1]],pos[e[1]])<D(pos[se[1]],pos[e[0]])):
				print('*')
				HC.append((se[0],e[0]))
				HC.append((se[1],e[1]))
			elif(D(pos[se[0]],pos[e[1]])<D(pos[se[0]],pos[e[0]]) and D(pos[se[1]],pos[e[0]])<D(pos[se[1]],pos[e[1]])):
				print('*')
				HC.append((se[0],e[1]))
				HC.append((se[1],e[0]))
			elif(D(pos[se[0]],pos[e[1]])<D(pos[se[0]],pos[e[0]]) and D(pos[se[1]],pos[e[1]])<D(pos[se[1]],pos[e[0]])):
				if(D(pos[se[0]],pos[e[1]])<D(pos[se[1]],pos[e[1]])):
					print('*')
					HC.append((se[0],e[0]))
					HC.append((se[1],e[1]))
				else:
					print('*')
					HC.append((se[0],e[1]))
					HC.append((se[1],e[0]))
					
			elif(D(pos[se[0]],pos[e[0]])<D(pos[se[0]],pos[e[1]]) and D(pos[se[1]],pos[e[0]])<D(pos[se[1]],pos[e[1]])):
				if(D(pos[se[0]],pos[e[0]])<D(pos[se[1]],pos[e[0]])):
					print('*')
					HC.append((se[0],e[1]))
					HC.append((se[1],e[0]))
				else:
					print('*')
					HC.append((se[0],e[0]))
					HC.append((se[1],e[1]))
			
			FPLs.append(HC)
			#break


FPLs.append(HC1)
print(len(FPLs))

pickle.dump(FPLs, open("FPLsConfig_VC_I=4_fullHCnobreak.txt", "wb"))
gr1=nx.Graph()
gr1.add_edges_from(FPLs[-1])

'''gr=nx.Graph()
for i in PLH:
	gr.add_edges_from(i)'''

nx.draw(gr1,pos,node_size=2, node_color='red',edge_color='purple',width=2) # ,with_labels=True,labels=LA)
#nx.draw(gr,pos,node_size=2, node_color='red',edge_color='red',width=1) # ,with_labels=True,labels=LA)

plt.show()
