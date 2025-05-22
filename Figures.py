import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import matplotlib.pylab as plt
import matplotlib as mt
import numpy as np
import pickle

import cairosvg

input_file = 'L31.svg'
output_file = 'output.pdf'

with open(input_file, 'rb') as f_in:
    svg_data = f_in.read()

with open(output_file, 'wb') as f_out:
    cairosvg.svg2pdf(bytestring=svg_data, write_to=f_out)


ghffh

# MDS Tiling............
'''
G = nx.read_gpickle('ContractedTilling_VC_I=4.gpickle')
pos = pickle.load(open('CO_NodesPositions_VC_I=4.txt', 'rb'))

G1 = nx.read_gpickle('../AB_HNew/ContractedTilling_ET_I=2.gpickle')
pos1 = pickle.load(open('../AB_HNew/CO_NodesPositions_ET_I=2.txt', 'rb'))

Pos1=dict((i,j*(1+np.sqrt(2))) for i,j in (pos1.items()))

N=list(G.nodes()).copy()
N1=list(G1.nodes()).copy()
for i in N:
	if(np.linalg.norm(pos[i]-pos[3])>25.5*np.sqrt(2)):
		G.remove_node(i)

for i in N:
	if(pos[i][0]<0 or pos[i][0]>26 or pos[i][1]<0 or pos[i][1]>26.8):
		G.remove_node(i)

for i in N1:
	if(Pos1[i][0]<0 or Pos1[i][0]>26.8 or Pos1[i][1]<0 or Pos1[i][1]>26.8):
		G1.remove_node(i)
	
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
	
nx.draw(G, pos = pos, node_size=0)
nx.draw(G1, pos = Pos1, node_size=8, node_color='red',width=2)

plt.show()
'''

# FF - fig1
'''
gr = nx.read_gpickle('../AB_HNew/ContractedTilling_ET_I=4.gpickle')
pos = pickle.load(open('../AB_HNew/CO_NodesPositions_ET_I=4.txt', 'rb'))
ED = pickle.load(open('../AB_HNew/DecorateEdge_ET_I=4.txt', 'rb'))

pos1 = pickle.load(open('../AB_HNew/CO_NodesPositions_ET_I=2.txt', 'rb'))
ED1 = pickle.load(open('../AB_HNew/DecorateEdge_ET_I=2.txt', 'rb'))

gr1d=nx.Graph()
gr1d.add_edges_from(ED1)
Pos1=dict((i,j*(1+np.sqrt(2))**2) for i,j in (pos1.items()))

SN = [(101,102),(102,61),(61,62),(62,21),(21,22),(22,10),(10,302),(302,261),(261,3),(3,221),(221,222),(222,181),(181,182),(182,141),(141,142),(142,101)]

gr2=nx.Graph()
gr2.add_edges_from(SN)
pos2=dict((i[0],pos1[i[0]]*(1+np.sqrt(2))**4) for i in SN)

W=[]
for u,v in list(gr.edges()):
	if((u,v) in ED or (v,u) in ED):
		W.append(0.7)	
	else:
		W.append(0)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')

nx.draw(gr1d, pos = Pos1, edge_color='red', node_size=0, width=4)
nx.draw(gr2, pos = pos2, edge_color='blue', node_size=0, width=5)
nx.draw(gr, pos = pos, node_size=0, width=W)

plt.xlim(-90, 90)
plt.ylim(-90, 90)
plt.show()
'''

# FF - fig2
'''
Pathsd1 = pickle.load(open('../AB_HNew/Paths_ET_I=4_d1_1.txt','rb')) 
Pathsd2 = pickle.load(open('../AB_HNew/Paths_ET_I=4_d1.txt','rb'))
gr = nx.read_gpickle('../AB_HNew/ContractedTilling_ET_I=4.gpickle')
pos = pickle.load(open('../AB_HNew/CO_NodesPositions_ET_I=4.txt', 'rb'))
ED = pickle.load(open('../AB_HNew/DecorateEdge_ET_I=4.txt', 'rb'))

P1=[]
for i in Pathsd1:
	P1.extend(i)

gr1=nx.Graph()
gr1.add_edges_from(P1)

P2=[]
for i in Pathsd2:
	P2.extend(i)
		
gr2=nx.Graph()
gr2.add_edges_from(P2)

EC=[]
W=[]
for u,v in list(gr.edges()):
	#if((u,v) in P1 or (v,u) in P1):
	#	W.append(1.3)
	#	EC.append('red')	
	if((u,v) in ED or (v,u) in ED):
		W.append(0.7)
		EC.append('black')
	else:
		W.append(0)
		EC.append('gray')

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')

nx.draw(gr1, pos = pos, node_size=0, edge_color='red', width=1.3)
nx.draw(gr2, pos = pos, node_size=0, edge_color='blue', width=1.3)
nx.draw(gr, pos = pos, node_size=0, edge_color=EC, width=W)


plt.xlim(-90, 90)
plt.ylim(-90, 90)
plt.show()
'''

# FF - fig3

Pathsd1 = pickle.load(open('../AB_HNew/Paths_ET_I=4_d1_1.txt','rb')) 
Pathsd2 = pickle.load(open('../AB_HNew/Paths_ET_I=4_d1.txt','rb'))
gr = nx.read_gpickle('../AB_HNew/ContractedTilling_ET_I=4.gpickle')
pos = pickle.load(open('../AB_HNew/CO_NodesPositions_ET_I=4.txt', 'rb'))
ED = pickle.load(open('../AB_HNew/DecorateEdge_ET_I=4.txt', 'rb'))

P1=[]
for i in Pathsd1:
	P1.extend(i)

P2=[]
for i in Pathsd2:
	P2.extend(i)

P3=P1.copy()
P4=P2.copy()
for u,v in P2:
	if((u,v) in P1):
		P3.remove((u,v))
		P4.remove((u,v))
	elif((v,u) in P1):
		P3.remove((v,u))
		P4.remove((v,u))

gr1=nx.Graph()
gr1.add_edges_from(P3)
		
gr2=nx.Graph()
gr2.add_edges_from(P4)

EC=[]
W=[]
for u,v in list(gr.edges()):
	#if((u,v) in P3 or (v,u) in P3):
	#	W.append(1.3)
	#	EC.append('red')	
	if((u,v) in ED or (v,u) in ED):
		W.append(0.7)
		EC.append('black')
	else:
		W.append(0)
		EC.append('gray')

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')

nx.draw(gr1, pos = pos, node_size=0, edge_color='red', width=1.3)
nx.draw(gr2, pos = pos, node_size=0, edge_color='blue', width=1.3)
nx.draw(gr, pos = pos, node_size=0, edge_color=EC, width=W)


plt.xlim(-90, 90)
plt.ylim(-90, 90)
plt.show()


# FF - fig4 - Giant H-cycle already there
'''
HC = pickle.load(open('../AB_HNew/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb')) 
pos = pickle.load(open('../AB_HNew/CO_NodesPositions_ET_I=4.txt', 'rb'))

gr=nx.Graph()
gr.add_edges_from(HC)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')

nx.draw(gr, pos = pos, node_size=0, width=0.7)

plt.xlim(-90, 90)
plt.ylim(-90, 90)
plt.show()
'''

# FF - fig5 - W0+W1+U0
'''
T = pickle.load(open("../AB_HNew/ContractedTilling_ET_I=2.gpickle",'rb'))
posT = pickle.load(open('../AB_HNew/CO_NodesPositions_ET_I=2.txt', 'rb'))

HC = pickle.load(open('../AB_HNew/Hcycle_d0+d1flips.txt','rb'))
posHC = pickle.load(open('../AB_HNew/CO_NodesPositions_VC_I=4.txt', 'rb'))

U1=nx.Graph()
U1.add_edges_from(HC)

pos = pickle.load(open('../AB_HNew/CO_NodesPositions_ET_I=2.txt', 'rb'))

SN0 = [(101,102),(102,61),(61,62),(62,21),(21,22),(22,10),(10,302),(302,261),(261,262),(262,221),(221,222),(222,181),(181,182),(182,141),(141,142),(142,101)]

SN1 = [(101,102),(102,61),(61,62),(62,21),(21,22),(22,10),(10,302),(302,261),(261,262),(262,221),(221,222),(222,181),(181,182),(182,141),(141,142),(142,101)]

W0=nx.Graph()
W0.add_edges_from(SN0)
pos0=dict((i[0],pos[i[0]]) for i in SN0)

W1=nx.Graph()
W1.add_edges_from(SN1)
pos1=dict((i[0],pos[i[0]]*(1+np.sqrt(2))**2) for i in SN1)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')

#nx.draw(W1, pos = pos1, node_size=0, edge_color='red', width=8) #,with_labels = True, labels=LA)
nx.draw(T,pos=posT, node_size=0, width=1)
nx.draw(U1, pos = posHC, edge_color='black', node_size=0, width=3)
#nx.draw(W0, pos = pos0, edge_color='orange',node_size=0, width=2)

plt.xlim(-90, 90)
plt.ylim(-90, 90)
plt.show()
'''

