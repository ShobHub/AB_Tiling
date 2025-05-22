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
	if(gr.degree(i)==8 or gr.degree(i)==7 or gr.degree(i)==6):
		NDom.append(i)

NDom.extend([724,364,4,5,2164,1804,1444,1084,727,726,367,366,7,6,2527,2526,2167,2166,1807,1806,1447,1446,1087,1086,9106,6615,1167,1166,6165,805,804,6225,807,6255,8746,5865,444,445,5805,806,407,5895,8386,5505,84,85,5445,446,447,86,87,5535,8476,8025,2604,2605,5475,7605,2606,2607,8055,10546,7665,2244,2245,1884,1885,7245,2246,2247,7695,10186,7305,1524,1525,6885,1886,1887,7335,9826,6945,9466,6585,1164,1165,6525,1526,1527,6975,1125,765,405,45,2565,2205,1845,1485])

C.extend(V)
Nodes=[3]
HC_N=[]
HC_ON=[]
for u,v in C:
	Nodes.append(u)
	Nodes.append(v)

for u,v in HC:
	HC_N.append(u)
	HC_N.append(v)

for u,v in HC_O:
	HC_ON.append(u)
	HC_ON.append(v)

Nodes.extend(HC_N)
Nodes.extend(HC_ON)
Nodes.extend([1125,765,405,45,2565,2205,1845,1485])

gr1=nx.read_gpickle('ContractedTilling_VC_I=4.gpickle')
for i in list(gr.nodes()):
	if(i not in Nodes):
		gr1.remove_node(i)

NDom_C=NDom.copy()
for i in NDom:
	if(i not in Nodes):
		NDom_C.remove(i)

Emap=[]
w1=[]
for u,v in list(gr1.edges()):
	if((u,v) in ED or (v,u) in ED):
		Emap.append('purple')
		w1.append(2)
	else:
		Emap.append('black')
		w1.append(1)

'''
print(len(list(gr1.nodes())))
with open(r'networks/NPgraph_E.txt', 'w') as fp:
    for item in list(gr1.edges()):
        fp.write("%s,%s\n" %(item[0],item[1]))
    print('Done')

a = fv.FVSFinder("NPgraph_E.txt",mode="maxcover")
print(a)

a = pickle.load(open("result/Minimal_FVSs.txt", "rb"))
print(a)
'''
print(len(list(gr1.nodes())))
print(nx.is_dominating_set(gr1, NDom_C))     #218, True
print(len(NDom_C))
CO=collections.Counter(NDom_C)
for i,j in CO.items():
	if(j>1):
		print(i)

'''
NDom_N=nx.dominating_set(gr, start_with=3)   #229
print(NDom_N)
print(len(NDom_N))


NDom_greedy = [18, 19, 26, 27, 1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99, 106, 107, 81, 83, 84, 85, 86, 87, 88, 89, 360, 361, 373, 392, 394, 364, 366, 367, 368, 369, 403, 422, 443, 444, 445, 446, 447, 448, 449, 720, 721, 724, 726, 727, 728, 729, 763, 782, 803, 804, 805, 806, 807, 808, 809, 1080, 1081, 1084, 1086, 1087, 1088, 1089, 1123, 1142, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1440, 1441, 1444, 1446, 1447, 1448, 1449, 1483, 1502, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1800, 1801, 1804, 1806, 1807, 1808, 1809, 1843, 1862, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 2160, 2161, 2164, 2166, 2167, 2168, 2169, 2203, 2222, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2521, 2526, 2527, 2528, 2529, 2563, 2582, 2604, 2605, 2606, 2607, 2608, 2609, 5448, 5444, 5445, 5447, 5474, 5475, 5477, 5528, 5505, 5507, 5558, 5535, 5537, 5804, 5805, 5807, 5837, 5889, 5865, 5883, 5886, 5887, 5919, 5895, 5913, 5916, 5917, 6164, 6165, 6167, 6197, 6249, 6225, 6243, 6246, 6247, 6279, 6255, 6273, 6276, 6277, 6524, 6525, 6527, 6557, 6609, 6585, 6603, 6606, 6607, 6639, 6615, 6633, 6636, 6637, 6884, 6885, 6887, 6917, 6969, 6945, 6963, 6966, 6967, 6999, 6975, 6993, 6996, 6997, 7244, 7245, 7247, 7277, 7329, 7305, 7323, 7326, 7327, 7359, 7335, 7353, 7356, 7357, 7604, 7605, 7607, 7637, 7689, 7665, 7683, 7686, 7687, 7719, 7695, 7713, 7716, 7717, 7967, 7997, 8049, 8025, 8043, 8046, 8047, 8079, 8055, 8073, 8076, 8077, 8386, 8387, 8476, 8477, 8746, 8747, 8837, 9106, 9107, 9197, 9466, 9467, 9557, 9826, 9827, 9917, 10186, 10187, 10277, 10546, 10547, 10637, 10907, 10997, 45, 50, 61]    # 273

#print(len(NDom_greedy))
#print(NDom_C)

NC=[]
w=[]
for i in list(gr1.nodes()):
	if(i in NDom_C):
		NC.append('red')
		w.append(20)
	else:
		NC.append('black')
		w.append(5)


nx.draw(gr1,pos=pos,edge_color=Emap,width=w1,node_size=w,node_color=NC)'''

'''
for node, (x, y) in pos.items():
	if(gr.degree(node)==5):
		text(x, y, node, fontsize=7, ha='center', va='center')'''

plt.show()