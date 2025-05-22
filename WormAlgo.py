#import matplotlib.pylab as plt
from matplotlib.pyplot import pause
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import pyplot
import networkx as nx    
import numpy as np
import collections as cl
import pickle
import random
import pylab
import time
import sys

sys.setrecursionlimit(5000)

#FPLs = pickle.load(open('../../Square/Worm/FPLsConfig_SQ64_1Mruns_100.txt','rb'))
pos = pickle.load(open("../../Square/SQ64_positions.txt","rb"))
PL_HC = pickle.load(open("../../Square/PlaquetteCycles1_SQ64_periodic.txt", "rb"))
HC = pickle.load(open('../../Square/SQ64_HC.txt','rb'))
'''
FPL = pickle.load(open('../I=4/ET/FPL_d0+d1+d2_3M.txt','rb'))
#FPLs = pickle.load(open('../I=4/Worm/FPLsConfigU2_worm_1Mruns_100.txt','rb'))
pos = pickle.load(open("../I=4/ET/CO_NodesPositions_ET_I=4.txt","rb"))
PL = pickle.load(open("../I=4/ET/PlaquetteCycles1_ET_I=4.txt", "rb"))
HC = pickle.load(open('../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))

pos = pickle.load(open("../I=6/CO_NodesPositions_ET_I=6.txt","rb"))
#PL = pickle.load(open("../I=6/PlaquetteCycles1_ET_I=6.txt", "rb"))
HC = pickle.load(open('../I=6/Hcycle_level=0_U3_loop.txt','rb'))

FPL = pickle.load(open('../I=2/FPL_d0+d1_2M.txt','rb'))
pos = pickle.load(open("../I=4/VC/CO_NodesPositions_VC_I=4.txt","rb"))
PL = pickle.load(open("../I=4/VC/PlaquetteCycles1_VC_I=4.txt", "rb"))
HC = pickle.load(open('../I=2/Hcycle_d0+d1flips.txt','rb'))
'''
HC_N=[]
for i in HC:
	if(i[0] not in HC_N):
		HC_N.append(i[0])
	if(i[1] not in HC_N):
		HC_N.append(i[1])
# PL_HC=[]
# E = []
# for i in PL:
# 	c=0
# 	for j in i:
# 		if(j[0] in HC_N):
# 			c=c+1
# 	if(c==4):
# 		PL_HC.append(i)
# 		E.extend(i)

E=[]
for i in PL_HC:
	E.extend(i)

A = HC.copy()
deg = dict((u,2) for u in HC_N)
def NotInA(u,B,v):
	for p,q in E:
		if((p==u or q==u) and (p!=v and q!=v)):                    
			if((p,q) not in B and (q,p) not in B):
				return (p,q)
	return (u,v)

def InA(u,B,v):
	if(u==v):
		v = list(set(dpts) - set([u]))[0]
	for i in B:
		if((i[0]==u or i[1]==u) and (i[0]!=v and i[1]!=v)):         
			return i  

def has_alternating_path(graph, B, start, end, visited, flag, P):
	if start == end:
		P.append(start)
		return True

	visited[start] = True
	for nn in graph.neighbors(start): 
		if not visited[nn] and ((start, nn) not in B and (nn, start) not in B) == flag :
			if has_alternating_path(graph, B, nn, end, visited, not flag, P):
				P.append(start)
				return True, P
	return False

def find_alternating_path(graph, B, start, end):
	visited = dict((I,False) for I in graph.nodes())
	if(deg[start] == 1):
		b = has_alternating_path(graph, B, start, end, visited, True, [])
		return b[1]
	elif(deg[start] == 3):
		b = has_alternating_path(graph, B, start, end, visited, False, [])
		return b[1]

def Flip_Path(PE,B):
	Bc = B.copy()
	for u,v in PE:
		if((u,v) in Bc):
			B.remove((u,v))
		elif((v,u) in Bc):
			B.remove((v,u))
		else:
			B.append((u,v))
	return B

g = nx.Graph()
g.add_edges_from(E)
x = random.choice(A)[0]
while(g.degree(x) == 2):
	x = random.choice(A)[0]
dpts = [x, x, -1]   

#pylab.ion()
Fpls=[]	
CF = 0      
d=3   
Ap=[] 
Mr = []
while(CF < 1000000):
	print(CF)
	Mr.append(int(np.linalg.norm(pos[dpts[0]] - pos[dpts[1]])))

	# NC = []
	# W=[]
	# G = nx.Graph()
	# G.add_edges_from(A)
	# for i in G.nodes():
	# 	if(i in dpts[:2]):
	# 		NC.append('red')
	# 		W.append(10)
	# 	elif(i == dpts[2]):
	# 		NC.append('green')
	# 		W.append(0)
	# 	else:
	# 		NC.append('steelblue')
	# 		W.append(5)
	#
	# #LA = dict((i,i) if i in dpts else (i,i) for i in G.nodes())
	# plt.clf()
	# nx.draw(G, pos=pos, node_size=W, node_color=NC) #, with_labels=True, labels=LA)
	# pause(0.08)
	# pylab.show()
	# plt.show()

	# if( CF%100 == 0 and (dpts[0] != dpts[1]) ):
	# 	P = find_alternating_path(g, A, dpts[0],dpts[1])
	# 	PE = [(P[I],P[I+1]) for I in range(len(P)-1)]
	# 	A = Flip_Path(PE,A)
	# 	deg[dpts[0]] = 2
	# 	deg[dpts[1]] = 2
	# 	dpts = [dpts[0], dpts[0], -1]
		
	if(dpts[0] == dpts[1]):
		print('*')
		Fpls.append(A.copy())
		if(g.degree(dpts[0]) == 2 and g.degree(dpts[1]) == 2):
			x = random.choice(A)[0]
			while(g.degree(x) == 2):
				x = random.choice(A)[0]
			dpts = [x,x,-1]
		p,q = NotInA(dpts[0],A,dpts[2])
		A.append((p,q))
		if(p == dpts[0]):
			dpts[2] = p   #.append(p)
			dpts[1] = q
		else:
			dpts[2] = q   #.append(q) 
			dpts[1] = p
		deg[dpts[0]] = 3
		deg[dpts[1]] = 3
	else:
		flag = False
		'''
		if(dpts[0] in g.neighbors(dpts[1])):
			if(g.degree(dpts[0]) == 1 and g.degree(dpts[1]) == 1):
				flag = True
				A.append((dpts[0],dpts[1]))
				deg[dpts[0]] = deg[dpts[0]]+1
				deg[dpts[1]] = deg[dpts[1]]+1
		'''
		a = random.choice([0,1])    
		if (deg[dpts[a]] == 3 and flag==False):                           
			b = InA(dpts[a],A,dpts[2])
			A.remove(b)

			if(b[0] == dpts[a]):
				dpts[2] = dpts[a]  #.append(dpts[a])  
				dpts[a] = b[1]
			
			else:
				dpts[2] = dpts[a]    #.append(dpts[a])
				dpts[a] = b[0]
		
			deg[b[0]] = deg[b[0]]-1
			deg[b[1]] = deg[b[1]]-1

		elif(deg[dpts[a]] == 1 and flag==False):
			while(d==3):
				p,q = NotInA(dpts[a],A, dpts[2])
				if(p == dpts[a]):
					dpts[2] = p   #.append(p) 
					dpts[a] = q
					d = deg[q]
				else:
					dpts[2] = q   #.append(q)
					dpts[a] = p
					d = deg[p]
			A.append((p,q))
			deg[p] = deg[p]+1
			deg[q] = deg[q]+1
			d=3
	CF=CF+1

print(len(Mr))
print(len(Fpls))
pickle.dump(Mr, open('../../Square/Worm/M(r)_SQ64_1Mruns_NR_.txt','wb'))
# pickle.dump(Fpls, open('../I=4/Worm/Test_1Mruns_noAltPath.txt','wb'))
# pickle.dump(Fpls, open('../../Square/Worm/FPLsConfig_worm_1Mruns.txt','wb'))       ## need to redo this file - overwritten by mistake on 24/08 1:27pm!!!!!

import pickle
import matplotlib.pylab as plt
import networkx as nx

# for i in Fpls[530:531]:
# 	g=nx.Graph()
# 	g.add_edges_from(i)
# 	#plt.clf()
# 	nx.draw(g, pos=pos, node_size=5)
# 	#pause(1)
# 	#pylab.show()
# 	plt.show()

import collections as cl
from scipy.optimize import curve_fit
from scipy.stats import linregress
import seaborn as sns
import numpy as np
#Mr = pickle.load(open('../../Square/Worm/M(r)_SQ64_1Mruns_NR_.txt','rb'))
Mr = pickle.load(open('../I=4/Worm/M(r)_1Mruns_NR.txt','rb'))
N = len(Mr)
MR = cl.Counter(Mr)
MR = {u:v/N for u,v in MR.items()}
MR = dict(sorted(MR.items()))
del MR[0.0]
#Mr=list(filter((0.0).__ne__, Mr))
def f(x, A, B):
	return A*x + B

#sns.histplot(Mr,bins=20,kde=True,stat='probability')
popt, pcov = curve_fit(f, np.log(list(MR.keys()))[1:11], np.log(list(MR.values()))[1:11])
print(popt)
s,_,_,_,se = linregress(np.log(list(MR.keys()))[1:11], np.log(list(MR.values()))[1:11])
print(s,se)
plt.plot(list(MR.keys())[1:11],np.exp(f(np.log(list(MR.keys()))[1:11],popt[0],popt[1])),'-r')
plt.plot(list(MR.keys())[1:11],np.exp(f(np.log(list(MR.keys()))[1:11],-0.33,popt[1])),'-b')
plt.scatter(MR.keys(),MR.values(),s=10)
plt.plot(MR.keys(),MR.values())
#plt.gca().invert_yaxis()
plt.xscale('log')
plt.yscale('log')
plt.show()































'''
Fpls = pickle.load(open('../I=4/Worm/Test_1Mruns_noAltPath.txt','rb'))

c=0
for i in Fpls[:]:
	print(c)
	G = nx.Graph()
	G.add_edges_from(i)
	c=c+1
	plt.clf()
	nx.draw(G, pos=pos, node_size=5)
	pause(5)
	pylab.show()
	#plt.show()
'''
'''
St = time.time() 
gr = nx.Graph()
gr.add_edges_from(B)
nB = g.edges()-gr.edges()
for p,q in nB:
	if((p==u or q==u) and (p!=v and q!=v)):                    
		return (p,q)
print(time.time()-St)
return (u,v)
############## Iteration Algo ##############################
def has_alternating_path(graph, B, start, end, flag):
    visited = set()
    parent = {}
    queue = deque()

    queue.append(start)
    visited.add(start)

    c=0
    while queue:
        current = queue.popleft()
        if(c!=0):
            flag = not parent[current][1]

        for neighbor in graph[current]:
            if neighbor not in visited and ((current, neighbor) not in B and (neighbor, current) not in B) == flag :
                visited.add(neighbor)
                parent[neighbor] = [current,flag]
                queue.append(neighbor)

                if neighbor == end:
                    path = [neighbor]
                    while current != start:
                        path.append(current)
                        current = parent[current][0]
                    path.append(start)
                    path.reverse()
                    return path
        c=c+1

    return []

def find_alternating_path(graph, B, start, end):
	if(deg[start] == 1):
		b = has_alternating_path(graph, B, start, end, True)
		return b
	elif(deg[start] == 3):
		b = has_alternating_path(graph, B, start, end, False)
		return b

#################

def has_alternating_path(graph, B, start, end, flag, P):
	visited = dict((I,False) for I in graph.nodes())
	queue = deque([(start, flag)])

	while queue:
		node, color = queue.popleft()
		P.append(node)

		if node == end:
			return P

		visited[node] = True

		for nn in graph.neighbors(node): 
			if not visited[nn] and ((node, nn) not in B and (nn, node) not in B) == flag :
				queue.append((nn, not flag))
	return False

def find_alternating_path(graph, B, start, end):
	if(deg[start] == 1):
		b=False
		while(b == False):
			b = has_alternating_path(graph, B, start, end, True, [])
		return b
	elif(deg[start] == 3):
		b=False
		while(b == False):
			b = has_alternating_path(graph, B, start, end, False, [])
		return b

		NC = []
		W=[]
		G = nx.Graph()
		G.add_edges_from(A)
		for i in G.nodes():
			if(i in dpts[:2]):
				NC.append('red')
				W.append(10)
			elif(i in P):
				NC.append('green')
				W.append(10)
			else:
				NC.append('steelblue')
				W.append(5)

		LA = dict((i,i) if i in P else (i,'') for i in G.nodes())	
		nx.draw(G, pos=pos, node_size=W, node_color=NC, with_labels=True, labels=LA)
		gg = nx.Graph()
		gg.add_edges_from(PE)
		nx.draw(gg, pos=pos, node_size = 0, width = 2, edge_color='red', alpha = 0.2)
		plt.show()
		
'''

