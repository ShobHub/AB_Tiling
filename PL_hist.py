import networkx as nx     
from scipy.optimize import curve_fit
import collections
import matplotlib.pylab as plt
#import seaborn as sns
import numpy as np
import pickle
from fractions import Fraction

## Total number of plaquettes in (VC,I=4) and (ET,I=4) = 7784,26808
## Total number of plaquettes in (HC,level 1) and (HC,level 2) = 381,14237


#FPLs=pickle.load(open("../AB_HNew/FPLsConfig_1M_d0+d1.txt", "rb"))
#pos = pickle.load(open("CO_NodesPositions_VC_I=4.txt","rb"))

L1=[]
PL = pickle.load(open("../AB_HNew/PL_1M_d0+d1_NewAlgo.txt", "rb"))
for i in PL[::380]:
	L1.extend(i) 

L=[]
PL0 = pickle.load(open("../AB_HNew/PL_1.5M_d0+d1+d2_NewAlgo.txt", "rb"))
#PL01 = pickle.load(open("../AB_HNew/PL_1.5Mon1.5M_d0+d1+d2_NewAlgo.txt", "rb"))
for i in PL0[:1000000:50]:
	L.extend(i) 
#for i in PL01[:1000000:50]:
#	L.extend(i) 

C=collections.Counter(L)
C1=collections.Counter(L1)
N = np.sum(list(C.values()))
N1 = np.sum(list(C1.values()))

Prob = {k : v / N for k, v in C.items() }
Prob1 = {k : v / N1 for k, v in C1.items() }

def f(x, A, B):
	return A*x + B

def RR(RG):
	x = sorted(list(RG.keys()))
	Y=[]
	X=[]
	for i in x:
		Y.append(np.log(RG[i]))
		X.append(np.log(i))
	return X,Y

X,Y = RR(Prob)
X1,Y1 = RR(Prob1)

popt, pcov = curve_fit(f, X[5:20], Y[5:20])   
popt1, pcov1 = curve_fit(f, X1[5:30], Y1[5:30])   
print(popt, popt1)


plt.axvline(x = 13.5, color = 'grey', label = 'Grey lines are log-periodic')
plt.axvline(x = 244, color = 'grey')
plt.axvline(x = 4787, color = 'grey')

#plt.scatter(X, Y, s=10, label = 'U1 {8_2} - FPL')
#plt.scatter(X1, Y1, s=10, label = 'U2 {8_4} - FPL')

plt.plot(list(Prob.keys()), list(Prob.values()), 'o', ms=4, label = 'Double_Inflation (U2)')
plt.plot(list(Prob1.keys()), list(Prob1.values()), 'o', ms=4, label = 'Single_Inflation (U1)')

#plt.plot(np.array(X[5:20]), f(np.array(X[5:20]), popt[0], popt[1]))
#plt.plot(np.array(X1[5:30]), f(np.array(X1[5:30]), popt1[0], popt1[1]))

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Loops Length (s)')
plt.ylabel('P(s), Probability of loops of length s')
plt.legend(fontsize=10)
plt.autoscale(True)
plt.xlim(1,10**5)
plt.ylim(10**(-12),10**6)
plt.show()












'''
from numpy.polynomial import Polynomial as P
from numpy.polynomial import Chebyshev as T

x = list(Prob1.keys())
y = list(Prob1.values())
logx = np.log10(x)
logy = np.log10(y)
poly = T.fit(logx,logy,30)   #np.polyfit(logx,logy,deg=20)
#poly = np.poly1d(coeffs)
yfit = lambda x: 10**(poly(np.log10(x)))
plt.plot(np.sort(x), yfit(np.sort(x)), c='blue') 



plt.scatter(list(Prob1.keys()),list(Prob1.values()), facecolors='none', edgecolors='b', label = 'Single_Inflation (U1)')

x1 = list(Prob.keys())
y1 = list(Prob.values())
logx1 = np.log10(x1)
logy1 = np.log10(y1)
poly1 = T.fit(logx1,logy1,30) #np.polyfit(logx1,logy1,deg=20)
#poly1 = np.poly1d(coeffs1)
yfit1 = lambda x1: 10**(poly1(np.log10(x1)))
plt.plot(np.sort(x1), yfit1(np.sort(x1)), c='green') 

plt.scatter(list(Prob.keys()),list(Prob.values()), facecolors='none', edgecolors='g', label = 'Double_Inflation (U2)')
'''


'''
HC=pickle.load(open("../AB_HNew/Hcycle_d0+d1flips.txt", "rb"))
g=nx.Graph()
g.add_edges_from(HC)

D=dict.fromkeys(list(g.nodes()),0)

for i in range(201):
	Loop = pickle.load(open("../AB_HNew/FPLsConfig_d0+d1/Loops_FPLs%s.txt"%i, "rb"))
	for j in Loop:
		l=len(j)
		for k in j:
			D[k]=D[k]+l
D={x:y/200 for x,y in D.items()}

values = list(D.values()) #[D.get(node, 0.25) for node in g.nodes()]

values = pickle.load(open("../AB_HNew/Values_NodeLoopLen.txt","rb"))

print(values)
print(len(values))

nx.draw(g, pos, node_size=100, node_color=values) #, with_labels=True, font_color='black', font_size=10)
N=nx.draw_networkx_nodes(g, pos, node_size=100, node_color=values) #, with_labels=True, font_color='black', font_size=10)
plt.colorbar(N)
plt.axis('off')
plt.show()


LA=dict.fromkeys(list(g.nodes()),'')
LA[3]='3'
D=[0]
F=list(range(100,10001,100))
for j in F:
	s=0
	for i in range(j-100,j):
		Loop = pickle.load(open("Loops_FPLs1/Loops_FPLs%s.txt"%i, "rb"))
		for k in Loop:
			if(3 in k):
				s=s+len(k)
	D.append((D[-1]*(j-100)+s)/j)

F=list(range(100,10001,100))
D = pickle.load(open("../AB_HNew/Node3vsFlips10k.txt","rb"))
print(D)
#plt.scatter(F,D[1:],label='central node',s=20)
plt.plot(F,D[1:])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('No.of Flips')
plt.ylabel('Average length of loops passing')
plt.legend()
plt.show()

'''



'''
PL = pickle.load(open("../AB_HNew/PL_475-575k_d0+d1.txt", "rb"))
c=0
z=[]
for i in PL: 
	c=c+1
	if(max(i)<100):
		z.append(c)
		print(max(i))

c=0
for i in z: #FPLs[475157:475158]:
	gr1=nx.Graph()
	gr1.add_edges_from(FPLs[i+475000])  #(i)
	nx.draw(gr1,pos,node_size=2, node_color='red',edge_color='purple',width=2) # ,with_labels=True,labels=LA)
	plt.savefig("../AB_HNew/FPLsConfig_d0+d1/FPLs_%s.png"%i)
	plt.clf()
	#plt.show()
	c=c+1

L1=[]
for i in range(1,19):
	print(25*i,25*i+25)
	PL = pickle.load(open("../AB_HNew/PL_%s-%sk_d0+d1.txt"%(25*i,25*i+25), "rb"))
	for i in PL:
		#print(max(i)) 
		L1.extend(i) #np.log10(i))

PL = pickle.load(open("../AB_HNew/PL_475-575k_d0+d1.txt", "rb"))
for i in PL: 
	L1.extend(i) #np.log10(i))

PL = pickle.load(open("../AB_HNew/PL_575-775k_d0+d1.txt", "rb"))
for i in PL: 
	L1.extend(i) #np.log10(i))
'''


	
