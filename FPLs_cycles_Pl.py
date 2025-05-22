import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
import collections
import matplotlib.pylab as plt
#import seaborn as sns
import numpy as np
import pickle

FPL = pickle.load(open("../AB_HNew/FPLsConfig_1M_d0+d1.txt", "rb"))
#FPL = pickle.load(open("FPLsConfig_noH_noLL_VC_I=4_200kFPLs.txt", "rb"))
print(len(FPL))

#FPL = pickle.load(open("FPLsConfig_SolidR_VC_I=4_5MNew.txt", "rb")
#FPL = pickle.load(open("FPLsConfig_VC_I=4_100kNew.txt", "rb"))
#FPL = pickle.load(open("FPLsConfig_CB30_1100x9.txt", "rb"))
#FPL1 = pickle.load(open("FPLsConfig_CB30_1100x2.txt", "rb"))
#FPL = pickle.load(open("FPLsConfig_new_RA_12.txt", "rb"))
#FPL = pickle.load(open("FPLsConfig_2nd-3rdloop_VC_I=4.txt", "rb"))
#FPL = pickle.load(open("FPLsConfig_1Inf-Oct+Soct.txt", "rb"))
#FPL = pickle.load(open("FPLsConfig_VCStar_I=4_50k.txt", "rb"))

#FPL=FPL+FPL1

def uniqueC(C):
	l=[len(i) for i in C]
	m=max(l)
	C1=[]
	for i in C:
		for j in range(m-len(i)):
			i.append('*')
		C1.append(i)
	C1=(np.unique(np.array(C1),axis=0)).tolist()
	C1=[list(set(i)-set(['*'])) for i in C1]
	return C1

L=[]
c=775000
for i in FPL[775000:]:
	gr=nx.Graph()
	gr.add_edges_from(i)
	gr=gr.to_directed()

	C=list(nx.simple_cycles(gr))
	C=[np.sort(j).tolist() for j in C]
	C1=uniqueC(C)
	c=c+1
	L1=[]
	for k in C1:
		if len(k)!=2:
			L1.append(len(k))
	L.append(L1)
	print(c)

#print(L)
pickle.dump(L, open("../AB_HNew/PL_775k-1M_d0+d1.txt", "wb"))
#pickle.dump(L, open("P(L)_CB30_7.txt", "wb"))
#pickle.dump(L, open("P(L)_0.txt", "wb"))
#pickle.dump(L, open("P(L)_2-3rdloop.txt", "wb"))
#pickle.dump(L, open("P(L)_1Inf-Oct+Soct.txt", "wb"))
#pickle.dump(L, open("P(L)_VCStar_I=4_50k_1.txt", "wb"))

skdjfshj

counter=collections.Counter(L)
#sns.displot(L, bins=50, kde=False)
plt.scatter(list(counter.keys()),list(counter.values()))
plt.xscale('log')
plt.yscale('log')
plt.ylabel('P(L) Loop Length Distribution')
plt.xlabel('Length of loops')

plt.show()

'''gr1=nx.Graph()
gr1.add_edges_from(FPL[-1])

nx.draw(gr,pos,node_size=2,node_color='red',edge_color='purple',width=1) # ,with_labels=True,labels=LA)'''

plt.show()

