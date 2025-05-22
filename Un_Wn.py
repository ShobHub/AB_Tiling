import networkx as nx     
import matplotlib.pylab as plt
import numpy as np
import pickle

print(16/33)

g1 = pickle.load(open('../I=2/ET/ContractedTilling_ET_I=2.gpickle','rb'))
pos1 = pickle.load(open('../I=2/ET/CO_NodesPositions_ET_I=2.txt','rb'))
hc1 = pickle.load(open('../I=2/Hcycle_d0+d1flips.txt','rb'))
pos_hc1 = pickle.load(open('../I=4/VC/CO_NodesPositions_VC_I=4.txt','rb'))
ghc1 = nx.Graph()
ghc1.add_edges_from(hc1)
print(len(ghc1.nodes()), len(g1.nodes()))
print(nx.transitivity(g1))

g2 = pickle.load(open('../I=4/ET/ContractedTilling_ET_I=4.gpickle','rb'))
pos2 = pickle.load(open('../I=4/ET/CO_NodesPositions_ET_I=4.txt','rb'))
hc2 = pickle.load(open('../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))
ghc2 = nx.Graph()
ghc2.add_edges_from(hc2)
print(len(ghc2.nodes()), len(g2.nodes()))
print(nx.transitivity(g2))
#
g3 = pickle.load(open('../I=6/ContractedTilling_ET_I=6.gpickle','rb'))
pos3 = pickle.load(open('../I=6/CO_NodesPositions_ET_I=6.txt','rb'))
hc3 = pickle.load(open('../I=6/Hcycle_level=0_U3_loop.txt','rb'))
ghc3 = nx.Graph()
ghc3.add_edges_from(hc3)
print(len(ghc3.nodes()), len(g3.nodes()))
# nx.draw(g1,pos=pos1,node_size=0,width = 0.3) #,with_labels=True)
# plt.savefig('../../../../../DownloadsFig1.pdf')
# #nx.draw(ghc3,pos=pos3,node_size=0,width = 1)
# plt.show()




