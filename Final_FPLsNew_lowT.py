import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import matplotlib.pylab as plt
import numpy as np
import copy
import pickle
import pylab
import time


FPL = pickle.load(open('../I=4/ET/FPL_d0+d1+d2_3M.txt','rb'))
pos = pickle.load(open("../I=4/ET/CO_NodesPositions_ET_I=4.txt","rb"))
PL = pickle.load(open("../I=4/ET/PlaquetteCycles1_ET_I=4.txt", "rb"))
HC = pickle.load(open('../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))

#HC=FPL[-1]

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

def flippable_plaq(HC):
	q='True'
	while(q=='True'):
		np.random.seed()
		n=np.random.randint(0,len(PL_HC))
		p=PL_HC[n]
		c=0
		ed=[]
		LR=[]
		for u,v in p:
			if((u,v) in HC):
				LR.append(p.index((u,v)))
				c=c+1
				ed.append(u)
				ed.append(v)
			elif((v,u) in HC):
				LR.append(p.index((u,v)))
				c=c+1
				ed.append(v)
				ed.append(u)

		if(c==2 and len(set(ed))==4):
			k=[(ed[0],ed[1]),(ed[2],ed[3])]
			q='False'
			if(LR==[0,2]):
				vl=1
				vr=0
			elif(LR==[1,3]):
				vl=0
				vr=1
		else:
			q='True'
	return k,vl,vr


def neigh(j,hc,vl,vr):
	a, b = j[0]
	c, d = j[1]
	cp=set([a,b,c,d])
	np=[]
	for k in PL_HC:
		pn = [k[0][0],k[1][0], k[2][0], k[3][0]] 
		l = len(cp.intersection(pn))
		if(l==2):
			np.append(k)
	J=1
	for k in np:
		c=0
		LR=[]
		ed=[]
		for u,v in k:
			if((u,v) in hc):
				LR.append(k.index((u,v)))
				c=c+1
				ed.append(u)
				ed.append(v)
			elif((v,u) in hc):
				LR.append(k.index((u,v)))
				c=c+1
				ed.append(v)
				ed.append(u)

		if(c==2 and len(set(ed))==4):
			J=J+1
			if(LR==[0,2]):
				vl=vl+1
			elif(LR==[1,3]):
				vr=vr+1
	E = -1*J+vl*5+vr*8
	return E

## keep values for vleg and vrung by multiplying it in vl,vr !! currently vleg =1 and vrung =1		

D = lambda posX, posY: np.linalg.norm(posX-posY)

FPLs=[]
FPLs.append(HC.copy())

nflips=0

Flip=[]
while(nflips<1000):
	f=[]
	i,vl,vr=flippable_plaq(HC)  
	vll=vl
	vrr=vr
	Ei = neigh(i,HC,vl,vr)

	se=i[0]
	e=i[1]
	
	#f.extend(i)
	
	#if(i[0] in HC and i[1] in HC):
	#nflips=nflips+1

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

	Ef = neigh(f,HC,vrr,vll)
	if(Ef-Ei<=0):
		#print(Ef-Ei,'*')
		nflips=nflips+1
		FPLs.append(HC.copy())
	else:
		prob = np.exp((Ei-Ef)/8)
		#print(Ef-Ei,prob,'**')
		if(np.random.uniform(0,1)<=prob):
			#print('#')
			nflips=nflips+1
			FPLs.append(HC.copy())
	#Flip.append(f)
	#FPLs.append(HC.copy())

print(len(FPLs))
pickle.dump(FPLs, open("../I=4/Worm/FPLsConfig_T=%.2f_V=-1_x.txt"%(8), "wb"))
#FPLs = pickle.load(open('../AB_HNew/FPLs_1M_d0+d1_T=1.txt', "rb"))
for i in FPLs[::-1][:10]:
	gr1=nx.Graph()
	gr1.add_edges_from(i)
	pylab.ion()
	plt.clf()
	nx.draw(gr1,pos,node_size=0, node_color='red',edge_color='purple',width=0.7) # ,with_labels=True,labels=LA)
	time.sleep(1)
	pylab.show()


