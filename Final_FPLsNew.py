import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import matplotlib.pylab as plt
import numpy as np
import copy
import pickle


PL = pickle.load(open("../I=4/ET/PlaquetteCycles1_ET_I=4.txt", "rb"))
HC = pickle.load(open("../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt", "rb"))
pos = pickle.load(open("../I=4/ET/CO_NodesPositions_ET_I=4.txt","rb"))

'''
Loops = pickle.load(open("Loops_Oct+Soct_2Bflips.txt", "rb"))

PathLN=[]
PathL = pickle.load(open("PathLoops_Oct+Soct_2Bflips.txt", "rb"))		
for u,v in PathL[0]:
	PathLN.append(u)
	PathLN.append(v)

L=Loops.copy()
for i in L:
	for u,v in i:
		if(u in PathLN or v in PathLN):
			Loops.remove(i)
			break
HC=[]
for i in Loops:
	HC.extend(i)
'''

#HC = FPL[-1]

HC_N=[]
for i in HC:
	HC_N.append(i[0])
	HC_N.append(i[1])

PL_HC=[]
for i in PL:
	c=0
	for j in i:
		if(j[0] in HC_N and j[1] in HC_N):
			c=c+1
	if(c==4):
		PL_HC.append(i)

print(len(PL_HC))

'''
AN=[]
AltP=[]
for i in PL_HC:
	n=[]
	c=0
	for u,v in i:
		n.append(u)
		if u not in AN:
			c=c+1
	if(c==4):
		AN.extend(n)
		AltP.extend(i)

gr=nx.Graph()
gr1=nx.Graph()
gr.add_edges_from(HC)
gr1.add_edges_from(AltP)
nx.draw(gr,pos,node_size=2)
nx.draw(gr1,pos,node_size=2, node_color='red',edge_color='purple',width=2)	
plt.show()
'''

def flippable_plaq(HC):
	q='True'
	while(q=='True'):
		np.random.seed()
		n=np.random.randint(0,len(PL_HC))
		p=PL_HC[n]
		c=0
		ed=[]
		for u,v in p:
			if((u,v) in HC):
				c=c+1
				ed.append(u)
				ed.append(v)
			elif((v,u) in HC):
				c=c+1
				ed.append(v)
				ed.append(u)

		if(c==2 and len(set(ed))==4):
			k=[(ed[0],ed[1]),(ed[2],ed[3])]
			q='False'
		else:
			q='True'
	return k
	

D = lambda posX, posY: np.linalg.norm(posX-posY)

FPLs=[]
FPLs.append(HC.copy())

nflips=0

Flip=[]
while(nflips<100000):
	print(nflips)
	f=[]
	i=flippable_plaq(HC)  

	se=i[0]
	e=i[1]
	
	f.extend(i)
	
	#if(i[0] in HC and i[1] in HC):
	nflips=nflips+1

	HC.remove(se)
	HC.remove(e)

	if(D(pos[se[0]],pos[e[0]])<D(pos[se[0]],pos[e[1]]) and D(pos[se[1]],pos[e[1]])<D(pos[se[1]],pos[e[0]])):
		HC.append((se[0],e[0]))
		HC.append((se[1],e[1]))
		f.append((se[0],e[0]))
		f.append((se[1],e[1]))
	
	elif(D(pos[se[0]],pos[e[1]])<D(pos[se[0]],pos[e[0]]) and D(pos[se[1]],pos[e[0]])<D(pos[se[1]],pos[e[1]])):
		HC.append((se[0],e[1]))
		HC.append((se[1],e[0]))
		f.append((se[0],e[1]))
		f.append((se[1],e[0]))

	elif(D(pos[se[0]],pos[e[1]])<D(pos[se[0]],pos[e[0]]) and D(pos[se[1]],pos[e[1]])<D(pos[se[1]],pos[e[0]])):
		if(D(pos[se[0]],pos[e[1]])<D(pos[se[1]],pos[e[1]])):
			HC.append((se[0],e[0]))
			HC.append((se[1],e[1]))
			f.append((se[0],e[0]))
			f.append((se[1],e[1]))

		else:
			HC.append((se[0],e[1]))
			HC.append((se[1],e[0]))
			f.append((se[0],e[1]))
			f.append((se[1],e[0]))
				
	elif(D(pos[se[0]],pos[e[0]])<D(pos[se[0]],pos[e[1]]) and D(pos[se[1]],pos[e[0]])<D(pos[se[1]],pos[e[1]])):
		if(D(pos[se[0]],pos[e[0]])<D(pos[se[1]],pos[e[0]])):
			HC.append((se[0],e[1]))
			HC.append((se[1],e[0]))
			f.append((se[0],e[1]))
			f.append((se[1],e[0]))

		else:
			HC.append((se[0],e[0]))
			HC.append((se[1],e[1]))
			f.append((se[0],e[0]))
			f.append((se[1],e[1]))

	#Flip.append(f)
	FPLs.append(HC.copy())

pickle.dump(FPLs[-1], open("../I=4/Worm/Test_WormvsFlip/FPL_flip_100kth.txt", "wb"))
print(FPLs[-1])
#FPLs = pickle.load(open('../AB_HNew/HeightRep/FPLsConfig_SQ30_1000_periodic.txt', "rb"))

'''
for i in FPLs:   #[::10000]:
	gr1=nx.Graph()
	gr1.add_edges_from(i)
	nx.draw(gr1,pos,node_size=2, node_color='red',edge_color='purple',width=2) # ,with_labels=True,labels=LA)
	plt.show()

'''
