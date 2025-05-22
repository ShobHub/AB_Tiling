import networkx as nx
from networkx import Graph, DiGraph, simple_cycles
from shapely.geometry import Point
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry.polygon import Polygon
import matplotlib.pylab as plt
from scipy.stats import linregress
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from itertools import chain
from collections import Counter
import pickle
import random
import time
import math

G = pickle.load(open('../I=4/ET/ContractedTilling_ET_I=4.gpickle','rb'))
FPLs = pickle.load(open('../I=4/Worm/FPLsConfig_T=8.00_V=-1.txt','rb'))
pos = pickle.load(open("../I=4/ET/CO_NodesPositions_ET_I=4.txt","rb"))
PL= pickle.load(open("../I=4/ET/PlaquetteCycles1_ET_I=4.txt", "rb"))
# HC = pickle.load(open('../I=2/Hcycle_d0+d1flips.txt','rb'))
HC = pickle.load(open('../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))
print(len(FPLs))

HC_N=[]
for i in HC:
	if(i[0] not in HC_N):
		HC_N.append(i[0])
	if(i[1] not in HC_N):
		HC_N.append(i[1])

PL_HC=[]
PL_HCN=[]
E=[]
for i in PL:
	c=0
	PN=[]
	for j in i:
		if(j[0] in HC_N and j[1] in HC_N):
			PN.append(j[0])
			PN.append(j[1])
			c=c+1
	if(c==4):
		PL_HCN.append(PN) #list(set(PN)))
		PL_HC.append(i)
		E.extend(i)
'''
Ord={}
c=0
for i in HC_N:
	Ord[c]=[]
	for j in PL_HCN:
		if(i in j):
			Ord[c].append(j)
	
	if(Ord[c]!=[]):
		PL_HCN = [e for e in PL_HCN if e not in Ord[c]]
		c=c+1
	else:
		c=c+0

Ord.popitem()

Ord1={}
for i,j in Ord.items():
	Ord1[i]=[]
	for k in j:
		m=[]
		for l in [0,2,4,6]:
			m.append((k[l],k[l+1]))
		Ord1[i].append(m)

po={}
for i,j in Ord.items():
	for k in j:
		pp = [pos[k[0]], pos[k[2]], pos[k[4]], pos[k[6]]]
		xc = sum([u[0] for u in pp])/4
		yc = sum([u[1] for u in pp])/4
		po[(xc,yc)] = i		


def PLAQ_H (pl, A, xs,ys):
	while(len(A) != 0):
		ce=[]
		c=0
		while(ce == [] and c<len(A)):
			pl1 = A[c]
			for u,v in pl1:
				if((u,v) in pl or (v,u) in pl):
					ce.append((u,v))
					break
			c=c+1
		if(ce!=[]):
			a,b = ce[0]
			pl1N = [pos[pl1[0][0]], pos[pl1[1][0]], pos[pl1[2][0]], pos[pl1[3][0]]]
			x = sum([i[0] for i in pl1N])/4
			y = sum([i[1] for i in pl1N])/4
	
			p=0
			for i in range(len(Loop)):
				if((a in Loop[i] and b in Loop[i]) and ((abs(Loop[i].index(a) - Loop[i].index(b))==1) or (abs(Loop[i].index(a) - Loop[i].index(b))== len(Loop[i])-1))):
					p=1
					d = D[i]
					point = Point(xs,ys)
					polygon = Polygon([pos[k] for k in Loop[i]])
					tf = polygon.contains(point)
					if((tf == True and d == 'C') or (tf == False and d == 'AC')):
						H[(x,y)] = H[(xs,ys)]+1
					elif((tf == True and d == 'AC') or (tf == False and d == 'C')):
						H[(x,y)] = H[(xs,ys)]-1
					break
			if(p == 0):
				H[(x,y)] = H[(xs,ys)]
	
			pl=pl1
			xs = x
			ys = y
			A.remove(pl1)
		
		else:
			Left.append(pl1)
			A.remove(pl1)
		
	return
	
def PLAQ_V (pl, i, x,y):
	ce=[]
	while(ce==[] and i>0):
		i = i-1
		for j in Ord1[i]:
			for u,v in j:
				if( ((u,v) in pl or (v,u) in pl) and j not in Left):
					plb = j
					ce.append((u,v))
					flag=True
					break
				else:
					flag = False
			if flag:
				break

	if(ce!=[]):
		a,b = ce[0]
		plbN = [pos[plb[0][0]], pos[plb[1][0]], pos[plb[2][0]], pos[plb[3][0]]]
		xs = sum([i[0] for i in plbN])/4
		ys = sum([i[1] for i in plbN])/4
	
		p=0
		for i in range(len(Loop)):
			if((a in Loop[i] and b in Loop[i]) and ((abs(Loop[i].index(a) - Loop[i].index(b))==1) or (abs(Loop[i].index(a) - Loop[i].index(b))== len(Loop[i])-1))):
				p=1
				d = D[i]
				point = Point(xs,ys)
				polygon = Polygon([pos[k] for k in Loop[i]])
				tf = polygon.contains(point)
				if((tf == True and d == 'C') or (tf == False and d == 'AC')):
					H[(x,y)] = H[(xs,ys)]+1
				elif((tf == True and d == 'AC') or (tf == False and d == 'C')):
					H[(x,y)] = H[(xs,ys)]-1
				break
		if(p == 0):
			H[(x,y)] = H[(xs,ys)]
	else:
		return('problem')
	
	return

H_FPLs = []
for q in range(len(FPLs)):
	print(q)
	Loop = pickle.load(open('../I=4/Worm/Loops_New/Loops_FPLs%i_T=8.00_V=-1.txt'%q,'rb'))
	#print(len(Loop))
	#lens = [len(i) for i in Loop]
	#m=lens.index(max(lens))
	#del Loop[m]
	#F = FPLs[q]     #DDLs[q-2000]  #*500]
	
	F=[]
	for i in Loop:
		for j in range(len(i)-1):
			F.append((i[j],i[j+1]))
		F.append((i[0],i[-1]))

	D=[]
	for i in range(len(Loop)):
		D.append(random.choice(['C','AC']))

	H={}
	Left=[]
	for i in range(len(list(Ord1.keys()))):
		A = Ord1[i].copy()
		pl = A[0]
		plN = [pl[0][0], pl[1][0], pl[2][0], pl[3][0]]
		x = sum([pos[i][0] for i in plN])/4
		y = sum([pos[i][1] for i in plN])/4

		if (i!=0):
			s = PLAQ_V(pl, i, *(x,y))
			c=0
			while(s=='problem'):
				c=c+1
				pl = A[c]
				plN = [pl[0][0], pl[1][0], pl[2][0], pl[3][0]]
				x = sum([pos[i][0] for i in plN])/4
				y = sum([pos[i][1] for i in plN])/4
				s=PLAQ_V(pl, i, *(x,y))		
			A.remove(pl)
			PLAQ_H (pl, A, *(x,y))
		else:
			A.remove(pl)
			H[(x,y)]=0
			PLAQ_H (pl, A, *(x,y))
	
	for l in Left:
		LP = l 
		LPN = [LP[0][0], LP[1][0], LP[2][0], LP[3][0]]
		x = sum([pos[i][0] for i in LPN])/4
		y = sum([pos[i][1] for i in LPN])/4
		for i in PL_HC:
			for u,v in LP:
				if( ((u,v) in i or (v,u) in i) and i not in Left):
					a,b=u,v
					plN = [i[0][0], i[1][0], i[2][0], i[3][0]]
					xs = sum([pos[i][0] for i in plN])/4
					ys = sum([pos[i][1] for i in plN])/4
					flag=True
					break
				else:
					flag=False
			if flag:
				break
		
		p=0
		for i in range(len(Loop)):
			if((a in Loop[i] and b in Loop[i]) and ((abs(Loop[i].index(a) - Loop[i].index(b))==1) or (abs(Loop[i].index(a) - Loop[i].index(b))== len(Loop[i])-1))):
				p=1
				d = D[i]
				point = Point(xs,ys)
				polygon = Polygon([pos[k] for k in Loop[i]])
				tf = polygon.contains(point)
				if((tf == True and d == 'C') or (tf == False and d == 'AC')):
					H[(x,y)] = H[(xs,ys)]+1
				elif((tf == True and d == 'AC') or (tf == False and d == 'C')):
					H[(x,y)] = H[(xs,ys)]-1
				break
		if(p == 0):
			H[(x,y)] = H[(xs,ys)]
	
	
	xo = 3.1525 
	yo = 8.54
	xxx=0
	for i in range(len(Loop)):
		if((6212 in Loop[i] and 5809 in Loop[i])):
			xxx=1
			d = D[i]
			point = Point(xo,yo)
			polygon = Polygon([pos[k] for k in Loop[i]])
			tf = polygon.contains(point)
			if((tf == True and d == 'C') or (tf == False and d == 'AC')):
				H['out'] = H[(xo, yo)]+1
			elif((tf == True and d == 'AC') or (tf == False and d == 'C')):
				H['out'] = H[(xo,yo)]-1
			break
	if(xxx==0):
		H['out'] = H[(xo,yo)]
		
	#print(H['out'])
	H_FPLs.append(H)
	#print(H)
	#print('************************')
	
	## PLOTTING ##

	# G = nx.Graph()
	# G.add_edges_from(E)
	#
	#
	# #LA={}
	# #PK=[]
	# #for i in Ord[0]:
	# #	PK.extend(i)
	# #for i in list(G.nodes()):
	# #	if(i in PK):
	# #		LA[i]=i
	# #	else:
	# #		LA[i]=''
	#
	#
	# EC=[]
	# W=[]
	# for u,v in list(G.edges()):
	# 	if ((u,v) in F or (v,u) in F):
	# 		EC.append('purple')
	# 		W.append(3)
	# 	else:
	# 		EC.append('gray')
	# 		W.append(0.8)

	# NC=[]
	# W1=[]
	# print(D[0])
	# for u in list(G.nodes()):
	# 	a='False'
	# 	for i in range(len(Loop)):
	# 		if (u in Loop[i]):
	# 			a=i
	# 			break
	# 	if(a!='False' and D[a] == 'C'):
	# 		NC.append('blue')
	# 		W1.append(25)
	# 	elif(a!='False' and D[a] == 'AC'):
	# 		NC.append('green')
	# 		W1.append(25)
	# 	else:
	# 		NC.append('yellow')
	# 		W1.append(0)

	# #nx.draw(G, pos = pos, node_size = 5, edge_color=EC, width = W)
	# nx.draw(G, pos = pos, node_size=W1, node_color=NC, edge_color=EC, width = W) #, with_labels=True) node_color=NC

	# for i,j in list(po.keys()):
	# 	plt.plot(i, j, marker="o", markersize=1, markeredgecolor="green", markerfacecolor="green")
	# 	plt.text(i,j-0.1,'%i'%po[(i,j)],c='g')

	# for i,j in list(H.keys())[:-1]:
	# 	plt.text(i,j,'%i'%H[(i,j)],c='r')
	# plt.show()

pickle.dump(H_FPLs, open('../I=4/Worm/Heights/Heights_FPLs_T=8.00_V=-1.txt','wb'))
fdsf
'''
################################ Continous FFT_2D  ######################

# pos = pickle.load(open('../../Penrose/ModifiedTiling/Cen_Pos_Mod_Tiling_(k,R)=(20,20).txt','rb'))
# PL_HC = pickle.load(open('../../Penrose/ModifiedTiling/Cen_PlaquetteCycles_(k,R)=(20,20).txt', 'rb'))
#H_FPLs = pickle.load(open('../../Penrose/ModifiedTiling/Heights_V=-0.037_T=0.02/Heights50_(k,R)=(10,10).txt','rb'))

'''
plaqs = []
poly_cen = []
for i in PL_HC:
	xc = 0
	yc = 0
	pp = []
	for u,v in i:
		xc = xc + pos[u][0]
		yc = yc + pos[u][1]
		xc = xc + pos[v][0]      #
		yc = yc + pos[v][1]      #
		#pp.append(pos[u])
		pp.extend([u,v])         #
	cen = (xc / (2 * len(i)), yc / (2 * len(i)))    #
	poly_cen.append(cen)         #
	pp = set(pp)                 #
	pp1 = []                     #
	for j in pp:                 #
		pp1.append(pos[j])       #
	pp1.sort(key=lambda a: math.atan2(a[1] - cen[1], a[0] - cen[0]))
	plaqs.append(Polygon(pp1))   #
	#plaqs.append(Polygon(pp))
	#poly_cen.append((xc/4,yc/4))
	#poly = Polygon(pp)
	#polygons[poly] = (xc/4,yc/4)

kdtree = KDTree(poly_cen)
def heights(x,y):
	point = Point(x,y)
	nearest_poly_ind = kdtree.query([x, y], k=10)[1]
	for I in nearest_poly_ind:
		if plaqs[I].contains(point) or point.touches(plaqs[I]):
			return poly_cen[I]
	return ('out')

for i in range(42,50):
	print(i)
	H = H_FPLs[i]
	x = np.linspace(-26,26,100)
	y = np.linspace(-26,26,90)
	H1 = np.zeros((len(y),len(x)))
	for j in range(len(y)):
		for k in range(len(x)):
			p = heights(x[k],y[j])
			H1[j][k] = H[p]
	pickle.dump(H1,open('../../Penrose/ModifiedTiling/Heights_V=-0.037_T=0.02/(k,R)=(10,10)/Heights_(-26,26,100,90)_%i.txt'%(i), 'wb'))
'''
def VarW(S,px,py):
	p = np.sqrt(np.square(py[:, np.newaxis]) + np.square(px))
	maxP = p.max()
	PM = []
	W2 = []
	l = np.concatenate((np.linspace(0.01,maxP/5,500),np.linspace(maxP/5,maxP,50)))
	for pm in l:
		w2 = 0
		rowI, colI = np.where(p<=pm)
		for u,v in zip(rowI,colI):
			w2 += S[u,v]
		W2.append(w2-S[0,0])
		PM.append(pm)
	return PM,W2

def heights(name,a,b,N,r):
	S=0
	for i in range(N):
		#print(i)
		H_2d = pickle.load(open(name%i, 'rb'))
		Hf = np.fft.fft2(H_2d)
		S = S + np.abs(Hf) ** 2
	S = S/(N*4*r**2)     #np.pi*r**2*N)
	#S[0,0] = 0
	#print(S.sum())
	px = np.fft.fftfreq(Hf.shape[1])[:a]
	py = np.fft.fftfreq(Hf.shape[0])[:b]
	PM, W2 = VarW(S,px,py)
	p = np.sqrt(np.square(py[:, np.newaxis]) + np.square(px))
	# S = 1/((p**2)*S[:b,:a])
	pd = list(np.diagonal(p))
	S1d = list(np.diagonal(S))[:b]

	p = list(np.ndarray.flatten(np.triu(p)))[:a]
	S = list(np.ndarray.flatten(np.triu(S[:b,:a])))[:a]
	return p,S,pd,S1d,PM,W2           #,PM,W2

# p,S = heights('../../Penrose/ModifiedTiling/Heights_V=-0.037_T=0.02/(k,R)=(10,10)/Heights_(-26,26,100,90)_%i.txt',50,45,48,26)
#p1,S1,PM1,W21 = heights('../I=2/Worm/Heights/HeightsU1_100,90_T=0.33/Heights_2D_(-13,13,100,90)_%i.txt',50,45,2,13)
#p2,S2,PM2,W22 = heights('../I=2/Worm/Heights/HeightsU1_100,90/Heights_2D_(-13,13,100,90)_%i.txt',50,45,4000,13)
#p,S = heights('../DDLs/Heights_2Darray_100,90_DDLs_U1_relabel/Heights_2D_(-13,13,100,90)_%i.txt', 50, 45,4000) 
#p3,S3 = heights('../DDLs/Heights_500,400_DDLs_U2/Heights_2D_(-76,76,500,400)_%i.txt', 250, 200,5000)
p,S,pd,S1d,PM,W2  = heights('../I=2/Heights/Heights_2Darray_100,90_U1/Heights_2D_(-13,13,100,90)_%i.txt', 50, 45, 3700, 13)  #, 0)
#p2,S2 = heights('../I=4/Heights/Heights_2Darray_100,90_U2/Heights_2D_(-76,76,100,90)_%i.txt', 50, 45, 2283,76, 0.002)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(1, 1, figsize=(104,104), sharex=True, sharey=True)   #105,105
def f(x, A, B):
    return A*x + B

plt.scatter(np.log(pd[1:]),-np.log(S1d[1:]), s=15000, facecolors='none', edgecolors='C0',linewidths=20, label='Data along (1,1)')   #3000, 10
plt.scatter(np.log(p[1:]),-np.log(S[1:]), s=15000,label='Data along (1,0)')

p.extend(pd[1:])
S.extend(S1d[1:])
s,intr,_,_,se = linregress(np.log(p[1:]), -np.log(S[1:]))
print(s,intr,se)
plt.plot(np.log(p[1:]),f(np.log(p[1:]),s-se,intr),label='Linear-Fit',c = 'gray', linewidth = 40)

x = np.arange(-4.5,0.5,1)
plt.plot(x,f(x,2,2), c = 'r', linewidth = 30, linestyle='--', label='Slope = 2')
'''
plt.scatter(p[1:],S[1:], s=2000,label='Data along (1,0)')

plt.plot(PM,W2)
plt.scatter(PM,W2,s=5,label='FPL_U1')
s,intr,_,_,se = linregress(np.log(PM[60:-50]), W2[60:-50])
print(s,intr,se)
plt.plot(PM[60:-50],f(np.log(PM[60:-50]),s,intr),label='Linear-Fit',c = 'gray', linewidth = 20)

H_ = pickle.load(open('../I=4/Worm/Heights/Heights_FPLs_T=8.00_V=-1.txt','rb'))    #Heights/Heights_FPLs1M_d0+d1+d2_500diff.txt', 'rb'))    #
df = pd.DataFrame(H_)
var_H = df.var(ddof=0).to_dict()
#pickle.dump(var_H,open('../I=4/Heights/VarH_FPLs1M_d0+d1+d2_500diff.txt', 'wb'))

cent = np.array((0,0))
DistC = {}
for c in list(H_[0].keys())[:-1]:
	d = np.round(np.linalg.norm(np.array(c) - cent),2)
	if d not in DistC.keys():
		DistC[d] = [c]
	else:
		DistC[d].append(c)
print(len(DistC.keys()),max(DistC)) #,DistC)

# L = list(np.linspace(7.5,8.9,5))+list(np.linspace(9,20,300))+list(np.linspace(20,70,50))   #25,10))
L = list(np.linspace(1,74,50))   #11
T = [0.3,2,5,7,8,10]
norm = mcolors.Normalize(vmin=min(T), vmax=max(T))   #[25:-25:2]  [0:-30:3]
cmap = cm.get_cmap('rainbow')
for i in range(len(T)):  #[0.1,0.3,0.5,0.8,1,2,3,4,1000]:
	var_H = pickle.load(open('../I=4/Worm/Heights/VarH_T=%.2f_V=-1.txt'%T[i], 'rb'))
	W = []
	for j in L:
		W.append(np.mean([var_H[q] for k, c in DistC.items() if k <= j for q in c]))
	plt.scatter(L[1:],W[1:],s=5000,label='T=%.2f'%T[i],color=cmap(norm(T[i])))
	plt.plot(L[1:],W[1:],linewidth = 20,color=cmap(norm(T[i])))

# s,intr,_,_,se = linregress(np.log(L[6:-10]), W[6:-10])
# print(s,intr,se)
# plt.plot(L[6:-10],f(np.log(L[6:-10]),s,intr),label='Linear-Fit',c = 'gray', linewidth = 20)

var_H = pickle.load(open('../I=4/Heights/VarH_FPLs1M_d0+d1+d2_500diff.txt', 'rb'))
W = []
for j in L:
	W.append(np.mean([var_H[q] for k, c in DistC.items() if k <= j for q in c]))
plt.scatter(L[1:],W[1:],s=5000,label='T=inf',c='brown')
plt.plot(L[1:],W[1:],linewidth = 20,c='brown')

plt.xscale('log')
'''
plt.ylabel(r'$-log(\langle |h(p)|^{2} \rangle)$', fontsize = 350)   #  Net Height Variance ($W^2$)   #400
plt.xlabel(r'$log(p)$',fontsize = 350)  #System Size ($L$)  400
plt.tick_params(axis='x', labelsize=300, length=90, width=20)   #350,80,20
plt.tick_params(axis='y', labelsize=300, length=90, width=20)   #350,80,20
plt.legend(fontsize = 300, loc='lower right')        # upper left
plt.savefig('XXXX.pdf')     #AB_varH_loops.pdf')
# plt.show()
dfg

'''
p=list(filter((0.0).__ne__, p))
S=list(filter((0.0).__ne__, S))
p.insert(0,0)
'''

########################## Real space coloring #####################################

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorsys

for t in [0.3,2,5,7,8,10,1000]:  #0.1,0.3,0.5,0.8,1,2,3,4,1000]:
	H = pickle.load(open('../I=4/Worm/Heights/VarH_T=%.2f_V=-1.txt' % t, 'rb'))
	# H = pickle.load(open('../I=4/Worm/Heights/Heights_FPLs_T=%.2f_V=-1.txt' % t, 'rb'))
	# df = pd.DataFrame(H)
	# H = df.mean().to_dict()

	hs = list(set(H.values()))
	coms = sorted(hs)
	mycols = {}
	part = 1.0 / len(coms)
	for k in range(len(coms)):
		mycols[coms[k]] = colorsys.hsv_to_rgb(50, 0.4, k * part)

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig, ax = plt.subplots()
	color_list = [mycols[val] for val in coms]
	norm = Normalize(vmin=min(coms), vmax=max(coms))
	sm = ScalarMappable(cmap=plt.cm.colors.ListedColormap(color_list), norm=norm)
	sm.set_array([])

	for i in PL_HC:
		plN = [pos[i[0][0]], pos[i[1][0]], pos[i[2][0]], pos[i[3][0]]]
		x = sum([k[0] for k in plN]) / 4
		y = sum([k[1] for k in plN]) / 4
		ax.add_patch(
			Polygon(plN, closed=True, fill=True, facecolor=mycols[H[(x, y)]], edgecolor='black', linewidth=0.15))

	cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.008)
	cbar.ax.tick_params(axis="y", labelsize=20, length=5, width=2)
	cbar.set_label('Height Variance',fontsize=20)
	ax.autoscale_view()
	ax.set_aspect('equal')
	ax.set_title('T=%.2f' % t, fontsize=20)
	ax.axis('off')
	plt.show()



	




