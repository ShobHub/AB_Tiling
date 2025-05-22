
import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import matplotlib.pylab as plt
import numpy as np
import pickle

pos = pickle.load(open('CO_NodesPositions_VC_I=4.txt', 'rb'))
FN = pickle.load(open("FPLsFlips_1M_d0+d1.txt", "rb"))

#HC = pickle.load(open("Hcycle_d0+d1flips.txt", "rb"))
HC = pickle.load(open("FPL_d0+d1+d2_1.5M_E.txt", "rb"))

FPL=[]
FPL.append(HC)

#FPL.extend(HC)    

#L=[]
#L.append(len(HC))
L=[len(k) for k in FPL]

PL=[]
PL.append(L)

for i in range(len(FN)):
	print(i)
	u,v = FN[i][0]
	u1,v1 = FN[i][1]

	for j in range(len(FPL)):
		if((u,v) in FPL[j] or (v,u) in FPL[j]):
			a=j
		if((u1,v1) in FPL[j] or (v1,u1) in FPL[j]):
			b=j

	if(a==b):
		l=PL[-1].copy()
		l.remove(len(FPL[a]))
		
		fp = FPL[a].copy()
		FPL.remove(FPL[a])

		if((u,v) in fp):
			fp.remove((u,v))
		else:
			fp.remove((v,u))

		if((u1,v1) in fp):
			fp.remove((u1,v1))
		else:
			fp.remove((v1,u1))
		
		fp.extend(FN[i][2:])
		
		G=nx.Graph()
		G.add_edges_from(fp)

		C=nx.find_cycle(G, source = u)
		C1=nx.find_cycle(G, source = v)
		
		FPL.append(C)
		FPL.append(C1)
		
		l.extend([len(C),len(C1)])
		PL.append(l)
	
	else:
		l=PL[-1].copy()
		l.remove(len(FPL[a]))
		l.remove(len(FPL[b]))
		l.append(len(FPL[a])+len(FPL[b]))
		PL.append(l)
		
		fp=FPL[a]+FPL[b]
		if((u,v) in fp):
			fp.remove((u,v))
		else:
			fp.remove((v,u))

		if((u1,v1) in fp):
			fp.remove((u1,v1))
		else:
			fp.remove((v1,u1))
		
		fp.extend(FN[i][2:])

		for index in sorted([a,b], reverse=True):
    			del FPL[index]
		
		FPL.append(fp)
	'''
	G1=nx.Graph()
	for k in FPL:
		G1.add_edges_from(k)
	LA=dict.fromkeys(list(G.nodes()),'')
	for k in FN[:25]:
		for x,y in k:
			LA[x]=x
			LA[y]=y
	nx.draw(G1,pos=pos,node_size=5) #,with_labels=True, labels=LA)
	plt.show()
	'''

#print(len(PL))
pickle.dump(PL, open("PL_1M_d0+d1_NewAlgo.txt","wb"))

'''LA=dict.fromkeys(list(G.nodes()),'')
for u,v in C1:
	LA[u]=u
	LA[v]=v
nx.draw(G,pos=pos,node_size=5,with_labels=True, labels=LA)
plt.show()'''








