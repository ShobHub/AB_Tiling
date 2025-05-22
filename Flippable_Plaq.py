import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import matplotlib.pylab as plt
import numpy as np
import pickle

FPLs = pickle.load(open("FPLsConfig_1M_d0+d1.txt","rb"))
pos = pickle.load(open("CO_NodesPositions_VC_I=4.txt","rb"))
PL= pickle.load(open("../AB Tiling New/PlaquetteCycles1_VC_I=4.txt", "rb"))

HC = FPLs[0]

HC_N=[]
for i in HC:
	HC_N.append(i[0])
	HC_N.append(i[1])

PL_HC=[]
E=[]
for i in PL:
	c=0
	for j in i:
		if(j[0] in HC_N and j[1] in HC_N):
			c=c+1
	if(c==4):
		E.extend(i)
		PL_HC.append(i)


def flippable_plaq(F):
	n=0
	FP=[]
	for j in PL_HC:
		ed=[]
		c=0
		for u,v in j:
			if((u,v) in F):
				c=c+1
				ed.append(u)
				ed.append(v)
			elif((v,u) in F):
				c=c+1
				ed.append(v)
				ed.append(u)

		if(c==2 and len(set(ed))==4):
			n=n+1
			FP.extend([(ed[0],ed[1]),(ed[2],ed[3])])
	return n,FP

def Loops(F):
	loop=[]
	L=[]
	ED_flip=F.copy()
	while(ED_flip!=[]):
		IV=ED_flip[0][0]
		p=ED_flip[0][1]
		L.append(ED_flip[0][0])
		ED_flip.remove(ED_flip[0])
		while(p!=IV):
			c=0
			for i in ED_flip:
				if(p==i[0]):
					c=c+1
					p=i[1]
					L.append(i[0])
					ED_flip.remove(i)
				elif(p==i[1]):
					c=c+1
					p=i[0]
					L.append(i[1])  #,i[0]))  #i
					ED_flip.remove(i)
			if(c==0):
				L=[]
				break
		if(c!=0):
			loop.append(L)
			L=[]

	return(len(loop))

for i in range(0,len(FPLs),20000):
	n,FP = flippable_plaq(FPLs[i])
	nL = Loops(FPLs[i])

	print(nL)
	print(i,n)
	#plt.scatter(i,n,c='steelblue')

	'''gr=nx.Graph()
	gr.add_edges_from(FPLs[i])
	nx.draw(gr,pos,node_size=2, node_color='',edge_color='purple',width=1)

	gr1=nx.Graph()
	gr1.add_edges_from(FP)
	nx.draw(gr1,pos,node_size=2, node_color='',edge_color='green',width=2)
	
	plt.savefig("Flip_plaqs_d0+d1/FPL_%d.png"%i)
	plt.close()'''
#plt.show()
	











