import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import matplotlib.pylab as plt
import numpy as np
import copy
import pickle

'''Flips = pickle.load(open("../AB_HNew/FPLsFlips_1.5M_d0+d1+d2.txt", "rb"))
HC = pickle.load(open("../AB_HNew/Hcycle_ET_I=4_d0+d1+d2flips.txt", "rb"))

with open('../AB_HNew/CO_NodesPositions_ET_I=4.txt', 'rb') as handle:
  pos = pickle.loads(handle.read())


#FPLs=[]
#FPLs.append(HC.copy())
c=0
for i in Flips:
	c=c+1
	print(c)
	HC.remove(i[0])
	HC.remove(i[1])
	HC.append(i[2])
	HC.append(i[3])
	#FPLs.append(HC.copy())
	if(c%10000==0):
		G=nx.Graph()
		G.add_edges_from(HC)
		nx.draw(G,pos,node_size=2, node_color='red')
		mng = plt.get_current_fig_manager()
		mng.full_screen_toggle()
		plt.show()
		#plt.savefig("../AB_HNew/FPLsConfig_d0+d1+d2/FPL_d0+d1+d2_%i.png"%c)
		#plt.close()'''

'''pickle.dump(HC, open("../AB_HNew/FPLsConfig_d0+d1+d2/FPL_d0+d1+d2_1.5M.txt", "wb"))

G=nx.Graph()
G.add_edges_from(HC)
nx.draw(G,pos,node_size=2, node_color='red')
plt.show()'''


