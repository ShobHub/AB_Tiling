import networkx as nx     
from networkx import Graph, DiGraph, simple_cycles
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy import sparse
import matplotlib.pylab as plt
import itertools
import numpy as np
import copy
import pickle

'''
#DV=pickle.load(open("../../Square/SQ64_Distance_Vertices.txt", "rb"))
DV=pickle.load(open("../I=4/ET/Distance_Vertices.txt", "rb"))

Cr={}
DVK=list(DV.keys())
print(len(DVK))

k=0
for i in DVK[:]: 
	print(k)
	v=DV[i]
	n=len(v)
	#print(n)
	s=0
	for j in range(73): #0,13000,50):
		C1 = pickle.load(open("../I=2/Worm/Loops_T=0.8/Loops_FPLs%i.txt"%j, "rb"))
		c=0
		for a,b in v:
			for p in C1:
				if(a in p and b in p):
					c=c+1
					break
		s=s+c/n
	Cr[i]=s/240   #260
	k=k+1

#pickle.dump(Cr, open("../I=2/C(r)_13000by50FPLs.txt","wb"))
'''

Cr = pickle.load(open("../I=2/Worm/C(r)_13000by50FPLs.txt", "rb"))
Cr2 = pickle.load(open("../I=4/Worm/C(r)_22379by112FPLs.txt", "rb"))

Cr = {key: value/1.2 for key, value in Cr.items()}

Cr = dict(sorted(Cr.items()))
Cr2 = dict(sorted(Cr2.items()))
X = np.log(list(Cr.keys()))
Y = np.log(list(Cr.values()))
X2 = np.log(list(Cr2.keys()))
Y2 = np.log(list(Cr2.values()))

def f(x, A, B):
    return A*x + B
    #return A * (x**(-B))

s,inter,_,_,se = linregress(X[1:-110],Y[1:-110])              # U1fplworm-[1:-115]    SQ64-[:-90]
print(s,inter,se)
s2,inter2,_,_,se2 = linregress(X2[:-80],Y2[:-80])             # U2fplworm-[:-80]
print(s2,inter2,se2)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(1, 1, figsize=(66,50), sharex=True, sharey=True)  #(78,78)

plt.scatter(list(Cr2.keys()),list(Cr2.values()), s=3000,label='U2 AB region, 14992 Nodes')
plt.plot(list(Cr2.keys()),list(Cr2.values()), linewidth =10)

plt.scatter(list(Cr.keys()), list(Cr.values()), s=3000,label='U1 AB region, 464 Nodes')
plt.plot(list(Cr.keys()),list(Cr.values()), linewidth =10)

# plt.plot(np.exp(np.array(X[1:-110])),np.exp(f(np.array(X[1:-110]),s,inter)),linewidth =20, c = 'gray')
plt.plot(np.exp(np.array(X2[:-80])),np.exp(f(np.array(X2[:-80]),s2,inter2)),linewidth =30, c = 'gray', label = 'Linear-Fit')
plt.xscale('log')
plt.yscale('log')

plt.ylabel(r'$P(s)$', fontsize = 210)
plt.xlabel(r'$s$',fontsize = 220)
plt.tick_params(axis='x', labelsize=175, length=60, width=10)
plt.tick_params(axis='y', labelsize=180, length=60, width=10)
plt.legend(fontsize = 160, loc='lower left')

plt.ylabel(r'$C(r)$')
plt.xlabel(r'$r$')
plt.savefig('Cr_AB.pdf')
plt.show()
						


