import networkx as nx
import matplotlib.pylab as plt
import collections as cl
from scipy.optimize import curve_fit
from matplotlib.pyplot import pause
import numpy as np
import pickle
import random
import pylab

class EWorm():
	def __init__(self,A,deg,dpts,T,V):
		self.A = A
		self.dpts = dpts
		self.deg = deg
		self.T = T
		self.V = V
	def plotAB(self,si,sj,sk):
		EC=[]
		W=[]
		for u,v in g.edges():
			if((u,v) in self.A or (v,u) in self.A):
				EC.append('purple')
				W.append(1)
			else:
				EC.append('black')
				W.append(0)
		NC = []
		w = []
		for u in g.nodes():
			if (u==si):
				NC.append('red')
				w.append(30)
			elif (u==sj):
				NC.append('red')
				w.append(30)
			else:
				NC.append('steelblue')
				w.append(5)
		pylab.ion()
		plt.clf()
		nx.draw(g,pos=pos,node_size=w,edge_color=EC,width=W,node_color=NC) #,with_labels=True,font_size=8)
		pause(0.001)
		pylab.show()
		#plt.show()
	def OP(self,u,v,flag,NV,NH):
		if (round(abs(pos[u][0] - pos[v][0]), 1) == 0.0 and flag == True):
			NV += 1
		elif (round(abs(pos[u][0] - pos[v][0]), 1) == 0.0 and flag == False):
			NV -= 1
		elif (round(abs(pos[u][1] - pos[v][1]), 1) == 0.0 and flag == True):
			NH += 1
		elif (round(abs(pos[u][1] - pos[v][1]), 1) == 0.0 and flag == False):
			NH -= 1
		return NV,NH
	def NfPQ(self,pls,m):
		NfP = 0
		for i in pls:
			c = 0
			ed = []
			for u, v in i:
				if ((u, v) in m or (v, u) in m):
					c = c + 1
					ed.append(u)
					ed.append(v)
			if (c == 2 and len(set(ed)) == 4):
				NfP += 1
		return NfP
	def plaqInv(self,u,v):
		pl = []
		for i in PL_HC:
			if((u,v) in i or (v,u) in i):
				pl.append(i)
		return pl
	def accept(self, NfP_i,NfP_f):
		kB = 1
		dE = self.V*(NfP_f - NfP_i)
		if(dE<0):
			return True
		else:
			prob = min(1, np.exp((-1*dE)/(kB * self.T)))
			u = random.random()
			if (u<prob):
				return True
			else:
				return False
	def NotInA(self,u, B, v):
		L = []
		for p, q in E:
			if ((p == u or q == u) and (p != v and q != v)):
				if ((p, q) not in B and (q, p) not in B):
					L.append((p, q))
		if(L!=[]):
			return L
		else:
			return [(u,v)]
	def InA(self,u, B, v):
		L = []
		if (u == v):
			v = list(set(dpts) - set([u]))[0]
		for i in B:
			if ((i[0] == u or i[1] == u) and (i[0] != v and i[1] != v)):
				L.append(i)
		return L
	def worm(self):
		Fpls = []
		CF = 0
		NH = 256 #3969
		NV = 0   #126
		Mr = []
		Nfp_i = self.NfPQ(PL_HC,self.A)
		nfp = []
		D=[]
		while (CF < 1000000):
			print(CF)
			# if(CF%100==0):
			# 	nfp.append(Nfp_i)
			# self.plotAB(dpts[0],dpts[1],dpts[2])
			# if(CF>90000):
			# 	Mr.append(int(np.linalg.norm(pos[self.dpts[0]] - pos[self.dpts[1]])))
			if (self.dpts[0] == self.dpts[1]):
				print('*')
				nfp.append(Nfp_i)
				D.append(abs(NV-NH)/256)   #3845)
				if(CF>0):
					Fpls.append(self.A.copy())
					# print(len(Fpls),'*')
				if (g.degree(self.dpts[0]) == 2 and g.degree(self.dpts[1]) == 2):
					x = random.choice(self.A)[0]
					while (g.degree(x) == 2):
						x = random.choice(self.A)[0]
					self.dpts = [x, x, -1]
				L = self.NotInA(self.dpts[0], self.A, self.dpts[2])
				for p,q in L:
					pls = self.plaqInv(p,q)
					fi = self.NfPQ(pls,self.A)
					A1 = self.A.copy()
					A1.append((p,q))
					ff = self.NfPQ(pls,A1)
					Nfp_f = Nfp_i - fi + ff
					if self.accept(Nfp_i, Nfp_f):
						break
				Nfp_i = Nfp_f
				self.A.append((p, q))
				NV,NH = self.OP(p,q,True,NV,NH)
				if (p == self.dpts[0]):
					self.dpts[2] = p  # .append(p)
					self.dpts[1] = q
				else:
					self.dpts[2] = q  # .append(q)
					self.dpts[1] = p
				self.deg[self.dpts[0]] = 3
				self.deg[self.dpts[1]] = 3
			else:
				a = random.choice([0, 1])
				if (self.deg[self.dpts[a]] == 3):
					L = self.InA(self.dpts[a], self.A, self.dpts[2])
					for p, q in L:
						pls = self.plaqInv(p, q)
						fi = self.NfPQ(pls, self.A)
						A1 = self.A.copy()
						A1.remove((p, q))
						ff = self.NfPQ(pls, A1)
						Nfp_f = Nfp_i - fi + ff
						if self.accept(Nfp_i, Nfp_f):
							break
					Nfp_i = Nfp_f
					self.A.remove((p,q))
					NV,NH = self.OP(p, q, False,NV,NH)
					b=(p,q)
					if (b[0] == self.dpts[a]):
						self.dpts[2] = self.dpts[a]  # .append(dpts[a])
						self.dpts[a] = b[1]
					else:
						self.dpts[2] = self.dpts[a]  # .append(dpts[a])
						self.dpts[a] = b[0]
					self.deg[b[0]] = self.deg[b[0]] - 1
					self.deg[b[1]] = self.deg[b[1]] - 1

				elif (self.deg[self.dpts[a]] == 1):
					L = self.NotInA(self.dpts[a], self.A, self.dpts[2])
					l=[]
					for p, q in L:
						if (p == self.dpts[a] and self.deg[q]!=3):
							l.append((p,q))
						elif (q == self.dpts[a] and self.deg[p]!=3):
							l.append((p,q))
					for p,q in l:
						pls = self.plaqInv(p, q)
						fi = self.NfPQ(pls, self.A)
						A1 = self.A.copy()
						A1.append((p, q))
						ff = self.NfPQ(pls, A1)
						Nfp_f = Nfp_i - fi + ff
						if self.accept(Nfp_i, Nfp_f):
							break
					Nfp_i = Nfp_f
					if (p == self.dpts[a]):
						self.dpts[2] = p
						self.dpts[a] = q
					else:
						self.dpts[2] = q
						self.dpts[a] = p
					NV,NH = self.OP(p, q, True,NV,NH)
					self.A.append((p, q))
					self.deg[p] = self.deg[p] + 1
					self.deg[q] = self.deg[q] + 1
			CF = CF + 1
		print(len(Fpls))
		return nfp,D,Mr,Fpls

FPLs = pickle.load(open('../I=4/Worm/FPLsConfigU2_worm_1Mruns_100.txt','rb'))
FPL = pickle.load(open('../I=4/ET/FPL_d0+d1+d2_3M.txt','rb'))
pos = pickle.load(open("../I=4/ET/CO_NodesPositions_ET_I=4.txt","rb"))
PL = pickle.load(open("../I=4/ET/PlaquetteCycles1_ET_I=4.txt", "rb"))
HC = pickle.load(open('../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))

# FPLs = pickle.load(open('../I=2/FPLsConfig_1M_d0+d1.txt','rb'))
# FPL = pickle.load(open('../I=2/FPL_d0+d1_2M.txt','rb'))
# pos = pickle.load(open("../I=4/VC/CO_NodesPositions_VC_I=4.txt","rb"))
# PL = pickle.load(open("../I=4/VC/PlaquetteCycles1_VC_I=4.txt", "rb"))
# HC = pickle.load(open('../I=2/Hcycle_d0+d1flips.txt','rb'))

# FPL = pickle.load(open('../../Square/FPL_maxFP_SQ16.txt','rb'))
# pos = pickle.load(open("../../Square/SQ16_positions.txt","rb"))
# PL_HC = pickle.load(open("../../Square/PlaquetteCycles1_SQ16_periodic.txt", "rb"))
# g = pickle.load(open('../../Square/SQ16.txt','rb'))
# E = list(g.edges())

HC_N=[]
for i in HC:
	if(i[0] not in HC_N):
		HC_N.append(i[0])
	if(i[1] not in HC_N):
		HC_N.append(i[1])
PL_HC = []
E = []
for i in PL:
	c=0
	for j in i:
		if(j[0] in HC_N):
			c=c+1
	if(c==4):
		PL_HC.append(i)
		for u,v in i:
			if((u,v) not in E and (v,u) not in E):
				E.append((u,v))
E=[]
for i in PL_HC:
	for u,v in i:
		if (u,v) not in E and (v,u) not in E:
			E.extend(i)
#print(E)

'''
A = FPL.copy()
deg = dict((u,2) for u in HC_N)
g = nx.Graph()
g.add_edges_from(E)
# nx.draw(g,pos,node_size=0)
# plt.show()
x = random.choice(A)[0]
while (g.degree(x) == 2):
	x = random.choice(A)[0]
dpts = [x, x, -1]

obj = EWorm(A,deg,dpts,T = 1,V = -1)
# obj.plotAB(0,0,0)
_,_,_,Fpls = obj.worm()
#Nfp = list(set(Nfp))
# obj.plotAB(0,0,0)

# plt.plot(list(range(len(Nfp))), Nfp,label='NfP,V=-1,T=1')
# plt.scatter(list(range(len(Nfp))), Nfp,s=5)
# plt.legend()
# plt.show()
# plt.plot(list(range(len(D))), D,label='D,V=-1,T=1')
# plt.scatter(list(range(len(D))), D,s=5)
# plt.legend()
# plt.show()
pickle.dump(Fpls,open('../I=2/Worm/FPLsConfig_T=1_V=-1.txt','wb'))

Fpls = pickle.load(open('../I=2/Worm/FPLsConfig_T=4.00_V=-1.txt','rb'))
print(len(Fpls))

for i in Fpls:
	g = nx.Graph()
	g.add_edges_from(i)
	pylab.ion()
	plt.clf()
	nx.draw(g,pos=pos,node_size=5)
	pause(0.5)
	pylab.show()
	#plt.show()
sdgfd
'''

Dt = []
T = []
MR = []
Nfpt = []
for k in [4]:  #0.1,0.3,0.5,0.8,1,2,3,4]:
	print(k)
	A = FPL.copy()   #HC
	deg = dict((u, 2) for u in HC_N)
	g = nx.Graph()
	g.add_edges_from(E)
	x = random.choice(A)[0]
	while (g.degree(x) == 2):
		x = random.choice(A)[0]
	dpts = [x, x, -1]

	obj = EWorm(A, deg, dpts, T=k, V=-1)
	_,_,_,Fpls = obj.worm()
	pickle.dump(Fpls, open('../I=4/Worm/FPLsConfig_T=%.2f_V=-1_2.txt'%k, 'wb'))
	# Dt.append(np.mean(D[-50:]))
	# Nfpt.append(np.mean(Nfp[-50:]))
	T.append(k)
	#MR.append(Mr)
	# print(D)
	# plt.plot(list(range(len(Nfp))), Nfp,label='NfP,V=-1,T=%0.2f'%k)
	# plt.scatter(list(range(len(Nfp))), Nfp,s=5)
	# plt.legend()
	# plt.show()
	# plt.plot(list(range(len(D))), D, label='D,V=-1,T=%0.2f'%k)
	# plt.scatter(list(range(len(D))), D, s=5)
	# plt.legend()
	# plt.show()

#0.1  2 1 3
#0.3  4 28 4  b
#0.5  8 4 0
#0.8  3 2 4
#1  101 4 7  b
#2  49 1  10  b
#3  1 5 4
#4  5 11 0

dgfd

# import pickle
# import collections as cl
# import matplotlib.pylab as plt
# import pickle
# L = pickle.load(open('../../Square/Worm/OP_DvsT_SQ64_200k.txt','rb'))
# Dt = L[1]
# T = L[0]
plt.plot(T,Nfpt,label='V=-1')
plt.scatter(T,Nfpt,s=5)
plt.legend()
plt.show()
plt.plot(T,Dt,label='V=-1')
plt.scatter(T,Dt,s=5)
plt.legend()
plt.show()
fdgff

pickle.dump([T,Nfpt], open('../I=2/Worm/OP_NFPvsT.txt','wb'))
pickle.dump(MR, open('../I=2/Worm/M(r)vsT.txt','wb'))
sfds

# pickle.dump([T,Dt], open('../../Square/Worm/OP_DvsT_SQ64_200k.txt','wb'))
# pickle.dump(MR, open('../../Square/Worm/M(r)vsT_SQ64_200k.txt','wb'))
#

import collections as cl
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import pickle
import numpy as np
#Ds = pickle.load(open('../../Square/Worm/OP_Ds_SQ64_500k.txt','rb'))
# L = pickle.load(open('../I=2/Worm/OP_NFPvsT.txt','rb'))
# MR = pickle.load(open('../I=2/Worm/M(r)vsT.txt','rb'))
# mr = pickle.load(open('../I=2/Worm/M(r)_2Mruns_NR.txt','rb'))
L = pickle.load(open('../../Square/Worm/OP_DvsT_SQ64_200k.txt','rb'))
MR = pickle.load(open('../../Square/Worm/M(r)vsT_SQ64_200k.txt','rb'))
mr = pickle.load(open('../../Square/Worm/M(r)_SQ64_1Mruns_NR.txt','rb'))
T = L[0]
def f(x,B,C,g):
	return B/(x**g)+C
g = []
beta = []
for k in range(4,len(T)-2,4):
	m = cl.Counter(MR[k])
	m = {u: v / 10000 for u, v in m.items()}
	m = dict(sorted(m.items()))
	del m[0.0]
	popt, pcov = curve_fit(f, np.log(list(m.keys()))[1:4],np.log(list(m.values()))[1:4])
	print(popt[2])
	g.append(popt[2])
	beta.append(1/k)
	plt.scatter(m.keys(), m.values(), s=10)
	plt.plot(m.keys(), m.values(),label='T=%0.02f'%T[k])
	plt.plot(list(m.keys())[1:4],np.exp(f(np.log(list(m.keys()))[1:4],popt[0],popt[1],popt[2])),'orange')
	# plt.scatter(list(range(len(Ds[k]))), Ds[k], s=5)
	# plt.plot(list(range(len(Ds[k]))), Ds[k], label='T=%0.02f' % T[k])
	# plt.legend()
	# plt.show()
m = cl.Counter(mr)
m = {u: v/10000 for u, v in m.items()}
m = dict(sorted(m.items()))
del m[0.0]
popt, pcov = curve_fit(f, np.log(list(m.keys()))[1:4],np.log(list(m.values()))[1:4])
print(popt[2])
g.append(popt[2])
beta.append(0.0)
plt.scatter(m.keys(), m.values(), s=10)
plt.plot(m.keys(), m.values(),label='T=inf')
plt.plot(list(m.keys())[1:4],np.exp(f(np.log(list(m.keys()))[1:4],popt[0],popt[1],popt[2])),'orange')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
plt.clf()
plt.scatter(beta,g,s=10)
plt.show()











