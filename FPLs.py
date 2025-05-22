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

#gr = nx.read_gpickle('ContractedTilling_VC_I=4.gpickle')
#with open('CO_NodesPositions_VC_I=4.txt', 'rb') as handle:
#  pos = pickle.loads(handle.read())
#ED_flip = pickle.load(open("EDflip_oct+soct_2BFlip.txt", "rb"))
#ED_flip = pickle.load(open("EDflip_oct+soct.txt", "rb"))
#ED_flip = pickle.load(open("../AB_HNew/EDafter_d0+d1+d2flips.txt", "rb"))
#BigP = pickle.load(open("BigFPLs_Paths_1Inf-oct+soct.txt","rb"))

FPL = pickle.load(open("../AB_HNew/FPL_d0+d1+d2_1.5M.txt", "rb"))

'''
X=BigP[2]
ED_flip=[]
for i in range(len(X)):
	if(i%2!=0):
		for u,v in X[i]:
			ED_flip.append((v,u))
	else:
		ED_flip.extend(X[i])
'''

loop=[]
L=[]
ED_flip=FPL.copy()
while(ED_flip!=[]):
	IV=ED_flip[0][0]
	p=ED_flip[0][1]
	L.append(ED_flip[0])            
	ED_flip.remove(ED_flip[0])
	while(p!=IV):
		c=0
		for i in ED_flip:
			if(p==i[0]):
				c=c+1
				p=i[1]
				L.append(i)
				ED_flip.remove(i)
			elif(p==i[1]):
				c=c+1
				p=i[0]
				L.append((i[1],i[0]))  #i
				ED_flip.remove(i)
		if(c==0):
			L=[]
			break
	if(c!=0):
		loop.append(L)
		L=[]

print(len(loop))

'''
Emap=[]
W=[]
loop1=[j for i in loop for j in i]
for u,v in gr.edges:
	if((u,v) in loop1 or (v,u) in loop1):
		Emap.append('black')
		W.append(1)
	else:
		Emap.append('white')
		W.append(0)'''

#pickle.dump(loop, open("../AB_HNew/LoopsED_ET_I=4_d0+d1+d2.txt", "wb"))
pickle.dump(loop, open("../AB_HNew/FPL_d0+d1+d2_1.5M_E.txt", "wb"))

#nx.draw(gr,pos,node_size=2, node_color='red',edge_color=Emap, width=W) # ,with_labels=True,labels=LA)

#plt.show()
