import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
from scipy.spatial import KDTree as KD
import matplotlib.pylab as plt
import geometry_sp as geo
import graph_mincopy as GC
import numpy as np
import pickle
import time
import sys

Pos = {3:0,2:0,1:0,0:0}
G = {2:0,1:0,0:0}
ED = {3:0,2:0,1:0,0:0}


#G[0] = nx.read_gpickle('../I=6/ContractedTilling_ET_I=6.gpickle') 
G[0] = pickle.load(open('../I=6/ContractedTilling_ET_I=6.gpickle','rb')) 
Pos[0] = pickle.load(open('../I=6/CO_NodesPositions_ET_I=6.txt', 'rb'))
ED[0] = pickle.load(open('../I=6/DecorateEdge_ET_I=6.txt', 'rb'))

'''
#G[1] = nx.read_gpickle('../I=4/ET/ContractedTilling_ET_I=4.gpickle')
Pos[1] = pickle.load(open('../I=4/ET/CO_NodesPositions_ET_I=4.txt', 'rb'))
ED[1] = pickle.load(open('../I=4/ET/DecorateEdge_ET_I=4.txt', 'rb'))
Pos[1]=dict((i,j*(1+np.sqrt(2))**2) for i,j in (Pos[1].items()))

G[2] = nx.read_gpickle('../I=2/ET/ContractedTilling_ET_I=2.gpickle')
Pos[2] = pickle.load(open('../I=2/ET/CO_NodesPositions_ET_I=2.txt', 'rb'))
ED[2] = pickle.load(open('../I=2/ET/DecorateEdge_ET_I=2.txt', 'rb'))
Pos[2]=dict((i,j*(1+np.sqrt(2))**4) for i,j in (Pos[2].items()))

ED[3] = [(101,102),(102,61),(61,62),(62,21),(21,22),(22,10),(10,302),(302,261),(261,3),(3,221),(221,222),(222,181),(181,182),(182,141),(141,142),(142,101)]
Pos[3]=dict((i[0],Pos[2][i[0]]*(1+np.sqrt(2))**2) for i in ED[3])

FL0 = nx.Graph()
FL0.add_edges_from(ED[0])

FL1 = nx.Graph()
FL1.add_edges_from(ED[1])

FL2 = nx.Graph()
FL2.add_edges_from(ED[2])

FL3 = nx.Graph()
FL3.add_edges_from(ED[3])
'''

#################################### Mapping 8s #############################################

#D = lambda posX,posY: np.linalg.norm(posX-posY)
'''
pos8_2={}
for i in list(G[2].nodes()):
	if(G[2].degree(i) == 8):
		pos8_2[i] = Pos[2][i]

Map3_2={}
for i in Pos[3].keys():
	for j in pos8_2.keys():
		if (D(Pos[3][i],pos8_2[j])<=1.215):
			Map3_2[i]=j


pos8_1={}
for i in list(G[1].nodes()):
	if(G[1].degree(i) == 8):
		pos8_1[i] = Pos[1][i]

Map2_1={}
for i in Pos[2].keys():
	for j in pos8_1.keys():
		if (D(Pos[2][i],pos8_1[j])<=1.215): #34.03):
			Map2_1[i]=j

pos8_0={}
for i in list(G[0].nodes()):
	if(G[0].degree(i) == 8):
		pos8_0[i] = Pos[0][i]

Map1_0={}
t = KD(list(pos8_0.values()))
for i in Pos[1].keys():
	a, b = t.query([Pos[1][i]], distance_upper_bound=5.85, k=1) 
	if(b[0]<len(pos8_0.keys())):
		Map1_0[i] = list(pos8_0.keys())[b[0]]

Maps = {3:Map3_2, 2:Map2_1, 1:Map1_0}

#pickle.dump(Maps, open('../Extras/Maps_HcycleGeneral_U3.txt', 'wb'))
'''
#Maps = pickle.load(open('../Extras/Maps_HcycleGeneral_U3.txt', 'rb'))


################################### Find 9-Paths #############################################
'''
CP = lambda posX,posY: posX[0]*posY[1]-posX[1]*posY[0]

def find_all_simple_paths(graph, src, end):
	current_paths = [[src]]
	
	while(len(current_paths[0]) < 10):
		next_paths = []
		if(len(current_paths[0]) == 9):
			for path in current_paths:
				for neighbor in graph.neighbors(path[-1]):
					if neighbor not in path and neighbor == end:
						new_path = path[:] + [neighbor]
						next_paths.append(new_path)
			current_paths = next_paths
		
		else:
			for path in current_paths:
				for neighbor in graph.neighbors(path[-1]):
					if neighbor not in path:
						new_path = path[:] + [neighbor]
						next_paths.append(new_path)
			current_paths = next_paths
	return current_paths
			

def findpath(a,b,q,G,DE,pos):

	Pnine = find_all_simple_paths(G, a, b)    #nx.all_simple_edge_paths(G, source=a, target=b, cutoff=9)

	P=[]
	for i in Pnine:           
		c=0
		CPV=[]
		for j in i:   
			#print( np.abs(np.round(CP(pos[j]-pos[a],pos[j]-pos[b]),1)) )
			if(np.abs(np.round(CP(pos[j]-pos[a],pos[j]-pos[b]),1))<=0.1):   #4.2): #46): #1.4):
				c=c+1
			else:
				CPV.append(np.round(CP(pos[j]-pos[a],pos[j]-pos[b]),2))
		#print(c)
		if(c==6 and all(k*q > 0 for k in CPV)):
			P.append(i) 

	P1 = []
	for i in P:
		p = []
		flag = True
		for j in [1,3,5,7]:   
			if((i[j],i[j+1]) in DE or (i[j+1],i[j]) in DE): 
				p.append( (i[j],i[j+1]) )
				continue
			else:
				flag = False
				break
		if(flag == True):
			for j in [0,2,4,6,8]:   
				if((i[j],i[j+1]) not in DE and (i[j+1],i[j]) not in DE): 
					p.insert( j, (i[j],i[j+1]) )
					continue
				else:
					flag = False
					break
			if(flag == True):
				P1.append(p)

	return P1


################################### All edges at level 0 for smallest 9-path #################

level = 1  #3
#E = ED[3]
E = pickle.load(open('../Extras/Hcycle_level=1_U3.txt', 'rb'))
'''
'''
while(level>0):
	N = len(E)
	print(N)
	Paths=[]
	for i in range(N):
		print(i)
		if(i%2==0):
			u,v = E[i]
			d=findpath(Maps[level][u],Maps[level][v],1,G[level-1],ED[level-1],Pos[level-1])  
			#print(d)  
			Paths.extend(d[0]) 
		else:
			u,v = E[i]
			d=findpath(Maps[level][u],Maps[level][v],-1,G[level-1],ED[level-1],Pos[level-1]) 
			#print(d)
			Paths.extend(d[0]) 

	pickle.dump(Paths, open('../Extras/Hcycle_level=0_U3_Paths.txt', 'wb'))
	E=ED[level-1].copy()
	for u,v in Paths:
		if((u,v) in ED[level-1]):
			E.remove((u,v))
		elif((v,u) in ED[level-1]):
			E.remove((v,u))
		else:
			E.append((u,v))
	
	pickle.dump(E, open('../Extras/Hcycle_level=0_U3.txt', 'wb'))
	
	level=level-1

	g = nx.Graph()
	g.add_edges_from(E)
	loop = nx.find_cycle(g,source=3)
	E = loop
	pickle.dump(E, open('../Extras/Hcycle_level=0_U3_loop.txt', 'wb'))
'''

E = pickle.load(open('../I=6/Hcycle_level=0_U3_loop.txt', 'rb'))
#print(E,len(E))

#nx.draw(G[0],pos=Pos[0],node_size=0, edge_color='grey') 
#nx.draw(G[1],pos=Pos[1],node_size=0, edge_color='red', width=2, with_labels=True, font_size=5)  #, labels=LA)
#nx.draw(FL2,pos=Pos[2],node_size=0, edge_color='blue', width=2) 
#nx.draw(FL3,pos=Pos[3],node_size=0, edge_color='green', width=2) 
#plt.show()

'''
for u,v in G[1].edges():
	if((u,v) in d[0]):
		EC.append('purple')
		W.append(3)
	elif((v,u) in d[0]):
		EC.append('purple')
		W.append(3)
	else:
		EC.append('black')
		W.append(1)
'''
plt.figure(figsize=(4,3.5)) #, dpi=100)
r=nx.Graph()
r.add_edges_from(E)
nx.draw(r,pos=Pos[0], node_size=0) 
plt.savefig('L3.pdf') #, dpi=100)
plt.show()



			
	

