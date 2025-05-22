import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import matplotlib.pylab as plt
from matplotlib.pyplot import figure, text
import collections
import fvsFinder as fv
import numpy as np
import pickle
import sys

gr = nx.read_gpickle('ContractedTilling_VC_I=4.gpickle')
ED = pickle.load(open("DecorateEdge_VC_I=4.txt", "rb"))
HC = pickle.load(open("HCycle_VC_I=4.txt", "rb"))
HC_O = pickle.load(open("HCycle_1Inf-Octagon_VC_I=4.txt", "rb"))
C = pickle.load(open("2nd-3rdloop_VC_I=4.txt", "rb"))
V = pickle.load(open("nearestStarloop_VC_I=4.txt", "rb"))
TL = pickle.load(open("3_loopsV_VC_I=4.txt", "rb"))

with open('CO_NodesPositions_VC_I=4.txt', 'rb') as handle:
  pos = pickle.loads(handle.read())


NDom=[]
for i in list(gr.nodes()):
	if(gr.degree(i)==8 or gr.degree(i)==7 or gr.degree(i)==6 or gr.degree(i)==5):
		NDom.append(i)
'''
Nodes = 

NDom_C=NDom.copy()
for i in NDom:
	if(i not in Nodes):
		NDom_C.remove(i)
'''

Emap=[]
w1=[]
for u,v in list(gr.edges()):
	if((u,v) in ED or (v,u) in ED):
		Emap.append('purple')
		w1.append(2)
	else:
		Emap.append('black')
		w1.append(1)

NC=[]
w=[]
for i in list(gr.nodes()):
	if(i in NDom):
		NC.append('red')
		w.append(20)
	else:
		NC.append('black')
		w.append(5)


nx.draw(gr,pos=pos,edge_color=Emap,width=w1,node_size=w,node_color=NC)

for node, (x, y) in pos.items():
	if(gr.degree(node)==5):
		text(x, y, node, fontsize=7, ha='center', va='center')

plt.show()