import networkx as nx    
from networkx import Graph, DiGraph, simple_cycles
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pylab as plt
from scipy.stats import linregress
import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit
from itertools import chain
import pickle
import random
import time

'''
#DDLs = pickle.load(open('../Josh Code/DDLs_MaxMatchU1_1M_relabel.txt','rb'))
FPLs = pickle.load(open('../I=4/Worm/FPLsConfigU2_worm_1Mruns_100.txt','rb'))
pos = pickle.load(open("../I=4/ET/CO_NodesPositions_ET_I=4.txt","rb"))
#PL= pickle.load(open("../I=4/VC/PlaquetteCycles1_VC_I=4.txt", "rb"))
PL= pickle.load(open("../I=4/ET/PlaquetteCycles1_ET_I=4.txt", "rb"))
#HC = pickle.load(open('../I=2/Hcycle_d0+d1flips.txt','rb'))
HC = pickle.load(open('../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))

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
				if((u,v) in pl or (v,u) in pl):
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
for q in range(0,1):
	print(q)
	Loop = pickle.load(open('../I=4/Worm/Loops/Loops_FPLs%i.txt'%q,'rb'))
	#lens = [len(i) for i in Loop]
	#m=lens.index(max(lens))
	#del Loop[m]
	F = FPLs[q]      #*500
	
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
			A.remove(pl)
			PLAQ_H (pl, A, *(x,y))
		else:
			A.remove(pl)
			H[(x,y)]=0
			PLAQ_H (pl, A, *(x,y))
	
	LP = Left[0]
	LPN = [LP[0][0], LP[1][0], LP[2][0], LP[3][0]]
	x = sum([pos[i][0] for i in LPN])/4
	y = sum([pos[i][1] for i in LPN])/4
	for i in PL_HC:
		for u,v in LP:
			if((u,v) in i or (v,u) in i):
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
	
	G = nx.Graph()
	G.add_edges_from(E)

	
	#LA={}
	#PK=[]
	#for i in Ord[0]:
	#	PK.extend(i)
	#for i in list(G.nodes()):
	#	if(i in PK):
	#		LA[i]=i
	#	else:
	#		LA[i]=''

	
	EC=[]
	W=[]
	for u,v in list(G.edges()):
		if ((u,v) in F or (v,u) in F):
			EC.append('purple')
			W.append(4)
		else:
			EC.append('black')
			W.append(1)
	
	NC=[]
	W1=[]
	for u in list(G.nodes()):
		for i in range(len(Loop)):
			if (u in Loop[i]):
				a=i
				break

		if(D[a] == 'C'):
			NC.append('blue')
			W1.append(25)
		elif(D[a] == 'AC'):
			NC.append('green')
			W1.append(25)
	
	#nx.draw(G, pos = pos, node_size = 5, edge_color=EC, width = W)
	nx.draw(G, pos = pos, node_size = W1, node_color=NC, edge_color=EC, width = W)

	#for i,j in list(po.keys())[0:2000]:
	#	plt.plot(i, j, marker="o", markersize=1, markeredgecolor="green", markerfacecolor="green")
	#	plt.text(i,j-0.1,'%i'%po[(i,j)],c='g')

	for i,j in list(H.keys())[:-1]:
		plt.text(i,j,'%i'%H[(i,j)],c='r')

	plt.show()
	
pickle.dump(H_FPLs, open('../I=2/Worm/Heights/Heights_FPLs_worm_1Mruns_2000.txt','wb'))
'''

############################### Continous FFT_2D  ######################

'''
H_FPLs = pickle.load(open('../I=2/Worm/Heights/Heights_FPLsU1_worm_1Mruns_2000-4000.txt','rb'))	

polygons = {}
for i in PL_HC:
	xc = 0
	yc = 0
	pp = []
	for u,v in i:
		xc = xc + pos[u][0]
		yc = yc + pos[u][1]
		pp.append(pos[u])
	poly = Polygon(pp)
	polygons[poly] = (xc/4,yc/4)

def heights(x,y):
	point = Point(x,y)
	for u,v in polygons.items():
		tf = u.contains(point)
		if(tf==True):
			return v
	return ('out')

for i in range(1500,2000):
	print(i)
	H = H_FPLs[i]

	x = np.linspace(-13,13,100)           #40)
	y = np.linspace(-13,13,90)           #30)
	H1 = np.zeros((len(y),len(x)))

	st = time.time()
	for j in range(len(y)):
		for k in range(len(x)):
			p = heights(x[k],y[j])
			
			#p1 = heights1(x[k],y[j])
			#if(p[0] == p1[0] and p[1] == p1[1]):
			#	print(True)
			#else:
			#	print(p,p1)
			#	print(False)
			
			H1[j][k] = H[p]

	pickle.dump(H1,open('../I=2/Worm/Heights/Heights_100,90_U1/Heights_2D_(-13,13,100,90)_%i.txt'%i, 'wb'))
'''

def heights(name, a, b, N, ss, ad):
	S=0
	for i in range(N):
		print(i)
		H_2d = pickle.load(open(name%i, 'rb'))
		Hf = np.fft.fft2(H_2d)
		S = S + np.abs(Hf) ** 2
	
	S = S/N

	px = np.fft.fftfreq(Hf.shape[1])[:a]        #[:50]
	#px = (px/max(px)) * (2*np.pi/ss) + ad
	
	py = np.fft.fftfreq(Hf.shape[0])[:b]          #[:45]
	#py = (py/max(py)) * (2*np.pi/ss) + ad


	S1 = np.zeros((len(py),len(px)))      #(45,50))
	p=np.zeros((len(py),len(px)))         #(45,50))
	for i in range(len(py)):
		for j in range(len(px)):
			p[i][j] = np.linalg.norm((px[j],py[i]))
			#S1[i][j] = ((2*(2-np.cos(px[j])-np.cos(py[i])))*S[i][j]) ** (-1)
			S1[i][j] = ((px[j]**2 + py[i]**2)*S[i][j]) ** (-1)
	
	
	pd = list(np.diagonal(p))
	S1d = list(np.diagonal(S))[:b]      #[:45]

	p = list(np.ndarray.flatten(np.triu(p)))[:a]      #[a+1:2*a]         #[:a]
	S = np.triu(S[:b,:a])                                    #[:45,:50])
	S = list(np.ndarray.flatten(S))[:a]        #[a+1:2*a]                  #

	p.extend(pd[1:])
	S.extend(S1d[1:])
	
	return p,S #,pd,S1d

#p1,S1,pd1,S1d1 = heights('../I=2/Worm/Heights/Heights_100,90_U1/Heights_2D_(-13,13,100,90)_%i.txt', 50, 45, 2000,13,0) 
p,S = heights('../DDLs/Heights_2Darray_100,90_DDLs_U1_relabel/Heights_2D_(-13,13,100,90)_%i.txt', 50, 45, 3999, 13,0) 
p3,S3 = heights('../DDLs/Heights_2Darray_100,90_DDLs_U2_relabel/Heights_2D_(-76,76,100,90)_%i.txt', 50, 45, 2213, 76,0.002)
#p1,S1,pd1,S1d1  = heights('../I=2/Heights/Heights_2Darray_100,90_U1/Heights_2D_(-13,13,100,90)_%i.txt', 50, 45, 0, 3700,13, 0)
#p2,S2,pd2,S1d2 = heights('../I=4/Heights/Heights_2Darray_100,90_U2/Heights_2D_(-76,76,100,90)_%i.txt', 50, 45, 1436,2283,76, 0.002)

#plt.scatter(np.ndarray.flatten(p),np.ndarray.flatten(S), label='DDL')
#plt.scatter(np.log(np.ndarray.flatten(p1)),-np.log(np.ndarray.flatten(S1)), label='FPL')

def f(x, A, B):
    return A*x + B

popt, pcov = curve_fit(f, np.log(p[1:]), -np.log(S[1:]))           #[1:50]  [1:20]
#popt1, pcov1 = curve_fit(f, np.log(p1[1:]), -np.log(S1[1:]))   
#popt2, pcov2 = curve_fit(f, np.log(p2[1:]), -np.log(S2[1:])*1.155)   
popt3, pcov3 = curve_fit(f, np.log(p3[1:]), -np.log(S3[1:]))       #*1.1)
print(popt,popt3)   

s,_,_,_,se = linregress(np.log(p[1:]), -np.log(S[1:]))
print(s,se)
#s,_,_,_,se = linregress(np.log(p1[1:]), -np.log(S1[1:]))
#print(s,se)
#s,_,_,_,se = linregress(np.log(p2[1:]), -np.log(S2[1:])*1.155)
#print(s,se)
s,_,_,_,se = linregress(np.log(p3[1:]), -np.log(S3[1:]))        #*1.1)
print(s,se)

plt.scatter(np.log(p[1:]),-np.log(S[1:]), s=10, c='darkorange', label='DDL_U1') 
#plt.scatter(np.log(pd[1:]),-np.log(S1d[1:]), s=10, c='darkorange') #, label='Data along (1,1) - DDL_U1')

#plt.scatter(np.log(p1[1:]),-np.log(S1[1:]), c='darkorange', s=10, label='FPL_U1')  #Data along (1,0) 
#plt.scatter(np.log(pd1[1:]),-np.log(S1d1[1:]), c='darkorange', s=10) #label='FPL_U1')  #Data along (1,1)

#plt.scatter(np.log(p2[1:]),-np.log(S2[1:])*1.155, s=10, c='blue', label='FPL_U2')  
#plt.scatter(np.log(pd2[1:]),-np.log(S1d2[1:])*1.155, s=10, c='blue') #, label='FPL_U2')

plt.scatter(np.log(p3[1:]),-np.log(S3[1:]), s=10, c='blue', label='DDL_U2')        #-np.log(S3[1:])*1.1
#plt.scatter(np.log(pd3[1:]),-np.log(S1d3[1:])*1.1, s=10, c='blue') #label='Data along (1,1) - DDL_U2')

plt.plot(np.log(p[1:]),f(np.log(p[1:]),popt[0],popt[1]),'orange')   #[1:50]  [1:20]
#plt.plot(np.log(p1[1:]),f(np.log(p1[1:]),popt1[0],popt1[1]),'orange') 
#plt.plot(np.log(p2[1:]),f(np.log(p2[1:]),popt2[0],popt2[1]),'blue') 
plt.plot(np.log(p3[1:]),f(np.log(p3[1:]),popt3[0],popt3[1]),'blue') 
'''

fig = plt.figure()

#plt.scatter(p,S,c='darkorange',label='- DDL_U1',s=10)  #Data along (1,0)
#plt.scatter(pd,S1d,c='darkorange',s=10)  #,label='- DDL_U1')    #Data along (1,1)

plt.scatter(p1,S1,c='r',s=10,label='- FPL_U1')
#plt.scatter(pd1,S1d1,c='r',s=10)  #,label='- FPL_U1')

#plt.scatter(p2,S2,c='g',s=10,label='- FPL_U2')
#plt.scatter(pd2,S1d2,c='g',s=10)    #,label='- FPL_U2')

#plt.scatter(p3,S3,c='b',label='- DDL_U2',s=10)
#plt.scatter(pd3,S1d3,c='b',s=10)  #,label='- DDL_U2',s=5)
'''
'''
ax = plt.gca()
ax.set_facecolor('xkcd:black')
fig.patch.set_facecolor('black')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.title.set_color('white')
'''

plt.ylabel(r'-log($<|h(p)|^{2}>$)', fontsize = 15)
plt.xlabel('log(p)',fontsize = 15)

#plt.ylim(-0.01,0.045)
#plt.ylabel(r'$1/(|p|^{2} <|h(p)|^{2}>)$',fontsize = 15)
#plt.xlabel('p', fontsize = 15)

#plt.ylabel(r'$<|h(p)|^{2}>$',fontsize = 15)
#plt.xlabel('p', fontsize = 15)

plt.legend(fontsize = 8, loc='upper left')
plt.show()

'''
p=list(filter((0.0).__ne__, p))
S=list(filter((0.0).__ne__, S))
p.insert(0,0)
'''

sdfdsd


########################## Real space coloring #####################################

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorsys

H_FPLs = pickle.load(open('../I=2/Heights/Heights_FPLs1M_d0+d1_500diff.txt','rb'))	

for k in range(5):

	H = H_FPLs[k]

	'''
	G = nx.Graph()
	G.add_edges_from(E)

	nx.draw(G, pos = pos, node_size = 5)

	for i,j in list(H.keys())[:-1]:
		plt.text(i,j,'%i'%H[(i,j)],c='r')

	plt.show()
	'''

	hs=list(set(H.values()))
	coms= sorted(hs)

	mycols = {}
	part = 1.0 / len(coms)
	for k in range(len(coms)): 
		mycols[coms[k]] = colorsys.hsv_to_rgb(50,0.4,k*part+0.15)
	
	fig = plt.figure()
	ax = plt.gca()
	for i in PL_HC:
		plN = [pos[i[0][0]], pos[i[1][0]], pos[i[2][0]], pos[i[3][0]]]
		x = sum([k[0] for k in plN])/4
		y = sum([k[1] for k in plN])/4
		
		ax.add_patch(Polygon(plN, closed=True, fill = True, facecolor = mycols[H[(x,y)]], edgecolor='black', linewidth = 0.15))

	
	ax.autoscale_view()
	'''
	ax.set_facecolor('xkcd:black')
	fig.patch.set_facecolor('black')
	ax.spines['bottom'].set_color('white')
	ax.spines['top'].set_color('white') 
	ax.spines['right'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.tick_params(axis='x', colors='white')
	ax.tick_params(axis='y', colors='white')
	ax.yaxis.label.set_color('white')
	ax.xaxis.label.set_color('white')
	ax.title.set_color('white')
	'''
	plt.gca().set_aspect('equal')
	plt.axis('off')
	plt.show()



'''
DDL = pickle.load(open('../Josh Code/DDLs_MaxMatch_1M_first250x4431.txt','rb'))
for i in DDL[0:200000:100]:
	g = nx.Graph()
	g.add_edges_from(i)
	nx.draw(g,pos=pos,node_size=5)
	plt.show()
	
dfdsfd
'''

############################### Discrete FFT_2D  ##############################
'''
import itertools as it

H_FPLs = pickle.load(open('Heights_FPLs1M_d0+d1_500diff.txt','rb'))	
N = len(list(H_FPLs[0].keys()))

px = np.arange(-10,10,0.5)
py = np.arange(-10,10,0.5)

P = list(it.product(px,py))
PX,PY = np.meshgrid(px,py)

HP_FPLs=np.zeros((len(H_FPLs),len(P))) #len(px)))

for q in range(200): #len(H_FPLs)):
	print(q)
	H = H_FPLs[q]
	del H['out']
	HP = []
	for i in range(len(P)): #len(px)):
		if(P[i]==(0.0,0.0)):
			c=i
		s=0
		for k,l in H.items():
			s = s + np.exp(1j*(P[i][0]*k[0]+P[i][1]*k[1])) * l
		s = s/np.sqrt(N)
		HP.append(np.abs(s)**2)
	HP_FPLs[q] = HP

HP2_M = np.mean(HP_FPLs, axis = 0)
p = [np.linalg.norm((u,v)) for u,v in P]      #[np.linalg.norm(i) for i in list(zip(px,py))]

HP2_M = list(HP2_M)
del HP2_M[c]
del p[c]

def f(x, A, B):
    return A*x + B

popt1, pcov1 = curve_fit(f, np.log(p), -np.log(HP2_M))
print(popt1)

plt.scatter(np.log(p), -np.log(HP2_M))
#plt.plot(np.log(p),f(np.log(p),popt1[0],popt1[1]),'-r')


plt.figure()
levels = np.linspace(0, 0.25, 25)
CS = plt.contourf(PX/np.pi, PY/np.pi, HP2_M.reshape(40,40), levels=levels, extend='min')
colorbar = plt.colorbar(CS)


plt.show()



############################### Continous FFT_2D  ######################

H_FPLs = pickle.load(open('Heights_FPLs1M_d0+d1_500diff.txt','rb'))	

def heights(y,x):
	point = Point(x,y)
	c=0
	for i in PL_HC:
		polygon = Polygon([pos[u] for u,v in i])
		tf = polygon.contains(point)
		if(tf==True):
			c=1
			xc = sum([pos[u][0] for u,v in i])/4
			yc = sum([pos[u][1] for u,v in i])/4
			break
	if(c==0):
		return ('out')
	else:
		return ((xc,yc))

def H_real(y,x,px,py,q):
	p = heights(y,x)
	IR = H_FPLs[q][p] * np.cos(px*x + py*y)
	return IR

def H_Imag(y,x,px,py,q):
	p = heights(y,x)
	II = H_FPLs[q][p] * np.sin(px*x + py*y)
	return II


px = np.arange(1,10,0.2)
py = np.arange(1,10,0.2)	

HP_FPLs=np.zeros((2,len(px)))  #len(H_FPLs)

for q in range(2): #len(H_FPLs)):
	print(q)
	print('*')
	H = H_FPLs[q]
	HP = []
	for i in range(len(px)):
		#s,e=integrate.dblquad(heights, -13, 13, -13, 13, args=(px[i],py[i],q))
		start = time.time()
		
		sr=integrate.nquad(H_real, [[-13,13],[-13,13]], args=(px[i],py[i],q), opts=[{'limit':1},{'limit':1}])
		si=integrate.nquad(H_Imag, [[-13,13],[-13,13]], args=(px[i],py[i],q), opts=[{'limit':1},{'limit':1}])
		
		print(f'Time: {time.time() - start}')
		s = sr[0] + 1j*si[0]
		#s = s/np.sqrt(N)
		HP.append(np.abs(s)**2)

	HP_FPLs[q] = HP

HP2_M = np.mean(HP_FPLs, axis = 0)
print(HP2_M.shape)
p = [np.linalg.norm(i) for i in list(zip(px,py))]
pickle.dump([HP2_M,p],open("Heights_log-log.txt",'wb'))
print(len(p))
plt.scatter(np.log(p), -np.log(HP2_M))
plt.show()

'''
	




