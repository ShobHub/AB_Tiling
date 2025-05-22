import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import matplotlib.pylab as plt
import geometry_sp as geo
import graph_mincopy as GC
import numpy as np
import pickle


gr = nx.read_gpickle('ContractedTilling_ET_I=4.gpickle')
pos = pickle.load(open('CO_NodesPositions_ET_I=4.txt', 'rb'))
ED = pickle.load(open('DecorateEdge_ET_I=4.txt', 'rb'))

gr1 = nx.read_gpickle('ContractedTilling_ET_I=2.gpickle')
pos1 = pickle.load(open('CO_NodesPositions_ET_I=2.txt', 'rb'))
ED1 = pickle.load(open('DecorateEdge_ET_I=2.txt', 'rb'))
DL = pickle.load(open('DecoLoops_ET_I=2.txt', 'rb'))

gr1d=nx.Graph()
gr1d.add_edges_from(ED1)
Pos1=dict((i,j*(1+np.sqrt(2))**2) for i,j in (pos1.items()))

SN = [(101,102),(102,61),(61,62),(62,21),(21,22),(22,10),(10,302),(302,261),(261,3),(3,221),(221,222),(222,181),(181,182),(182,141),(141,142),(142,101)]

gr2=nx.Graph()
gr2.add_edges_from(SN)
pos2=dict((i[0],pos1[i[0]]*(1+np.sqrt(2))**4) for i in SN)

'''
pos8_1={}
LA={}
for i in list(gr1.nodes()):
	if(gr1.degree(i) == 8):
		pos8_1[i] = Pos1[i]
		LA[i]=i
	else:
		LA[i]=''

Map1={}
LA1={}
for i in pos2.keys():
	LA1[i]=i
	for j in pos8_1.keys():
		if (np.linalg.norm(pos2[i]-pos8_1[j])<=0.25):
			Map1[i]=j
			#print(i,j)

pos8={}
for i in list(gr.nodes()):
	if(gr.degree(i) == 8):
		pos8[i] = pos[i]

Map={}
for i in Pos1.keys():
	for j in pos8.keys():
		if (np.linalg.norm(Pos1[i]-pos8[j])<=0.065):
			Map[i]=j


D = lambda posX,posY: np.linalg.norm(posX-posY)
CP = lambda posX,posY: posX[0]*posY[1]-posX[1]*posY[0]

def getpath(u,v,X):
	for i in X: #Paths:
		if (i[0][0]==u and i[8][1]==v or i[0][0]==v and i[8][1]==u):
			return i 

def findpath(a,b,q,G,DE,pos):
	PN=nx.all_simple_edge_paths(G, source=a, target=b, cutoff=9)
	Pnine=[]
	for i in list(PN):
		if(len(i)==9):
			Pnine.append(i)

	C=[]
	for i in Pnine:
		c=0
		C1=[]
		for u,v in i:
			if((u,v) in DE or (v,u) in DE):
				C1.append(c)
			c=c+1
		C.append(C1)

	P1=[]
	for i in range(len(C)):
		if(C[i]==[1,3,5,7]):
			P1.append(Pnine[i])
	
	P3=[]
	for i in P1:
		P2=[]
		for u,v in i:
			if(u not in P2):
				P2.append(u)
			if(v not in P2):
				P2.append(v)
		P3.append(P2)

	P4=[]
	for i in range(len(P3)):
		c=0
		CPV=[]
		for j in P3[i]:
			if(np.abs(np.round(CP(pos[j]-pos[a],pos[j]-pos[b]),1))<=0.1):
				c=c+1
			else:
				CPV.append(np.round(CP(pos[j]-pos[a],pos[j]-pos[b]),2))
		if(c==6 and all(k*q > 0 for k in CPV)):
			P4.append(P1[i])

	return P4

#____________________ PATHS d2 ___________________________________

Paths1=[]
for i in range(len(SN)):
	u,v=SN[i] 
	if(i%2==0):
		print(u,v)
		d=findpath(Map1[u],Map1[v],1,gr1,ED1,pos1)     
		Paths1.append(d[0])
	else:
		d=findpath(Map1[u],Map1[v],-1,gr1,ED1,pos1)     
		Paths1.append(d[0])
			

pickle.dump(Paths1,open('Paths_ET_I=4_d2.txt','wb'))

#____________________ PATHS d2 ___________________________________

#____________________ PATHS d1 ___________________________________


Paths1 = pickle.load(open('Paths_ET_I=4_d2.txt','rb'))

P1=[]
for i in Paths1:
	P1.extend(i)

Paths=[]
for i in range(len(Paths1)):
	if(i%2==0):
		for j in range(len(Paths1[i])):
			u,v=Paths1[i][j] 
			if(j%2==0):
				d=findpath(Map[u],Map[v],-1,gr,ED,pos)     
				Paths.append(d[0])
			else:
				d=findpath(Map[u],Map[v],1,gr,ED,pos)     
				Paths.append(d[0])
	else:
		for j in range(len(Paths1[i])):
			u,v=Paths1[i][j] 
			if(j%2==0):
				d=findpath(Map[u],Map[v],1,gr,ED,pos)     
				Paths.append(d[0])
			else:
				d=findpath(Map[u],Map[v],-1,gr,ED,pos)     
				Paths.append(d[0])
			

pickle.dump(Paths,open('Paths_ET_I=4_d1.txt','wb'))

Paths = pickle.load(open('Paths_ET_I=4_d1.txt','rb'))

D=DL.copy()
for i in range(len(D)):
	if(i in [20,17,15,13,11,9,7,3]):
		D1=[]
		for j in D[i]:
			D1.append((j[1],j[0]))
		DL.remove(D[i])
		DL.append(D1)

PathsF=[]
I1=[]
for i in DL:
	I=[0]*len(i)
	for j in range(len(i)):
		u,v=i[j]
		if((u,v) in P1 or (v,u) in P1):
			I[j]=1
			PathsF.append(getpath(Map[u],Map[v],Paths))
	I1.append(I)


for i in range(len(I1)):
	i1=I1[i].copy()
	j=np.array(I1[i])
	ij = np.where(j==1)[0]
	print(ij)
	if(len(ij)!=0):   #(i!=1 and i!=2) and 
		print(i)
		l=ij[0]
		for k in range(l):
			d=findpath(Map[DL[i][l-1][0]],Map[DL[i][l-1][1]],(-1)**k,gr,ED,pos)     
			PathsF.append(d[0])
			l=l-1
		for p in range(len(ij)-1):
			l=ij[p+1]-ij[p]
			for k in range(1,l):
				d=findpath(Map[DL[i][k+ij[p]][0]],Map[DL[i][k+ij[p]][1]],(-1)**(k+1),gr,ED,pos)     
				PathsF.append(d[0])
		l=ij[-1]
		for k in range(len(I1[i])-l-1):
			d=findpath(Map[DL[i][l+1][0]],Map[DL[i][l+1][1]],(-1)**k,gr,ED,pos)     
			PathsF.append(d[0])
			l=l+1
			
pickle.dump(PathsF,open('Paths_ET_I=4_d1_1.txt','wb'))

#____________________ PATHS d1 ___________________________________

PathsF = pickle.load(open('Paths_ET_I=4_d1_1.txt','rb'))
PathsF1 = pickle.load(open('Paths_ET_I=4_d1.txt','rb'))

P=[]
for i in PathsF:
	P.extend(i)

P1=[]
for i in PathsF1:
	P1.extend(i)

EDF=ED.copy()
for u,v in P:
	if((u,v) in EDF):
		ED.remove((u,v))
	elif((v,u) in EDF):
		ED.remove((v,u))
	else:
		ED.append((u,v))
EDF=ED.copy()
for u,v in P1:
	if((u,v) in EDF):
		ED.remove((u,v))
	elif((v,u) in EDF):
		ED.remove((v,u))
	else:
		ED.append((u,v))

pickle.dump(ED,open("EDafter_d0+d1+d2flips.txt","wb"))

g=nx.Graph()
g.add_edges_from(ED)
nx.draw(g,pos,node_size=5)

EC=[]
W=[]
for u,v in list(gr.edges()):
	if((u,v) in P or (v,u) in P):
		EC.append('red')
		W.append(2)	
	else:
		EC.append('black')
		W.append(1)

D=[]
for i in DL:
	D.extend(i)
EC1=[]
LA={}
for u,v in list(gr1d.edges()):
	if((u,v) in D or (v,u) in D):
		LA[u]=u
		LA[v]=v
		EC1.append('purple')
	else:
		LA[u]=''
		LA[v]=''
		EC1.append('black')

nx.draw(gr, pos = pos, node_size=5, edge_color=EC, width=W) #,with_labels = True, labels=LA)
nx.draw(gr1d, pos = Pos1, edge_color=EC1, node_size=5, width=2, with_labels = True, labels=LA)
nx.draw(gr2, pos = pos2, edge_color='blue', node_size=5, width=3) #, with_labels = True, labels=LA1)

loops = pickle.load(open("LoopsED_ET_I=4_d0+d1+d2.txt", "rb"))

le=[]
for i in loops:
	le.append(len(i))

m=loops[le.index(max(le))]   #len = 14992
print(len(m))

#pickle.dump(m,open("Hcycle_ET_I=4_d0+d1+d2flips.txt","wb"))
'''

m = pickle.load(open('Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))
g=nx.Graph()
g.add_edges_from(m)
nx.draw(g,pos,node_size=5)

plt.show()

'''
# Starts here Rough....
DLN=[]
for i in DL[:3]:
	print(i)
	for u,v in i:
		if(u not in DLN):
			DLN.append(u)
		if(v not in DLN):
			DLN.append(v)
LA={}
for i in list(gr1.nodes()):
	if( i in DLN):
		LA[i]=i
	else:	
		LA[i]=''     # Till here Rough....
'''
