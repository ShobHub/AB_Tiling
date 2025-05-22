import networkx as nx
import collections as cl
import matplotlib.pylab as plt
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
			if((u,v) in e or (v,u) in e):
				EC.append((1,0,0,0.5))
				W.append(5)
			elif((u,v) in self.A or (v,u) in self.A):
				EC.append('purple')
				W.append(2)
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
				NC.append('blue')
				w.append(30)
			elif (u==sk):
				NC.append('green')
				w.append(30)
			else:
				NC.append('steelblue')
				w.append(5)
		pylab.ion()
		plt.clf()
		nx.draw(g,pos=pos,node_size=w,edge_color=EC,width=W,node_color=NC) #,with_labels=True,font_size=8)
		pause(0.5)
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
		dE = self.V * (NfP_f - NfP_i)
		prob = min(1, np.exp((-1*dE)/(kB * self.T)))
		n = random.random()
		if (n<prob):
			return True
		else:
			return False
	def NotInA(self, u, B):
		L = []
		for p, q in E:
			if (p == u or q == u):
				if ((p, q) not in B and (q, p) not in B):
					L.append((p, q))
		return L
	def InA(self, u, B):
		L = []
		for i in B:
			if (i[0] == u or i[1] == u):
				L.append(i)
		return L
	def worm(self):
		Fpls = []
		CF = 0
		nfp = []
		NH = 4096
		NV = 0
		Nfp_i = self.NfPQ(PL_HC,self.A)
		D = []
		Mr = []
		while (CF < NR):
			print(CF)
			if(4 in self.deg.values()):
				print('True')
			nfp.append(Nfp_i)
			D.append(abs(NV - NH) / 4096)
			if(CF > NR-10000):
				Mr.append(round(np.linalg.norm(pos[self.dpts[0]] - pos[self.dpts[1]]),1))
			#self.plotAB(self.dpts[0],self.dpts[1],self.dpts[2])
			if (self.dpts[0] == self.dpts[1]):
				mark = 0
				Fpls.append(self.A.copy())
				x = random.choice(self.A)[0]
				while (g.degree(x) == 2):
					x = random.choice(self.A)[0]
				self.dpts = [x, x, -1]
				L = self.InA(self.dpts[1], self.A)
				p,q = random.choice(L)
				pls = self.plaqInv(p,q)
				fi = self.NfPQ(pls,self.A)
				A1 = self.A.copy()
				A1.remove((p,q))
				ff = self.NfPQ(pls,A1)
				Nfp_f = Nfp_i - fi + ff
				if self.accept(Nfp_i, Nfp_f):
					Nfp_i = Nfp_f
					NV, NH = self.OP(p, q, False, NV, NH)
					self.A.remove((p, q))
					if (p == self.dpts[0]):
						self.dpts[2] = p
						self.dpts[1] = q
					else:
						self.dpts[2] = q
						self.dpts[1] = p
					self.deg[self.dpts[0]] = 1
					self.deg[self.dpts[1]] = 1
			else:
				a = 1
				if (self.deg[self.dpts[a]] == 1):
					L = self.NotInA(self.dpts[a], self.A)
					l = 0
					for p, q in L:
						if ((p == self.dpts[a] and q == self.dpts[0]) or (p == self.dpts[0] and q == self.dpts[a])) and mark != 0:
							pls = self.plaqInv(p, q)
							fi = self.NfPQ(pls, self.A)
							A1 = self.A.copy()
							A1.append((p, q))
							ff = self.NfPQ(pls, A1)
							Nfp_f = Nfp_i - fi + ff
							if self.accept(Nfp_i, Nfp_f):
								Nfp_i = Nfp_f
								self.dpts[2] = self.dpts[a]
								self.dpts[1] = self.dpts[0]
								self.A.append((p, q))
								self.deg[p] = self.deg[p] + 1
								self.deg[q] = self.deg[q] + 1
								NV, NH = self.OP(p, q, True, NV, NH)
							else:
								pls = self.plaqInv(self.dpts[a], self.dpts[2])
								fi = self.NfPQ(pls, self.A)
								self.A.append((self.dpts[a], self.dpts[2]))
								ff = self.NfPQ(pls, self.A)
								Nfp_i = Nfp_i - fi + ff
								self.deg[self.dpts[2]] = self.deg[self.dpts[2]] + 1
								self.deg[self.dpts[a]] = self.deg[self.dpts[a]] + 1
								self.dpts = [self.dpts[0], self.dpts[2], self.dpts[a]]
								NV, NH = self.OP(self.dpts[a], self.dpts[2], True, NV, NH)
							l = 1
							break
					if l==0:
						W = {}
						nf = {}
						for p, q in L:
							pls = self.plaqInv(p, q)
							fi = self.NfPQ(pls, self.A)
							A1 = self.A.copy()
							A1.append((p, q))
							ff = self.NfPQ(pls, A1)
							Nfp_f = Nfp_i - fi + ff
							W[(p, q)] = np.exp(-self.V * (ff) / self.T)
							nf[(p, q)] = Nfp_f
						Z = np.sum(list(W.values()))
						minn = min(list(W.values()))
						den = Z - minn
						Pr = {}
						n = random.random()
						for i, j in W.items():
							if (i == (self.dpts[a], self.dpts[2]) or i == (self.dpts[2], self.dpts[a])) and mark!=0:
								Pr[i] = (j - minn) / den
							elif (i == (self.dpts[a], self.dpts[2]) or i == (self.dpts[2], self.dpts[a])) and mark == 0:
								Pr[i] = 0
							else:
								Pr[i] = j / den
							if(n < Pr[i]):
								p, q = i  #max(Pr, key=Pr.get)
								Nfp_i = nf[(p,q)]
								NV, NH = self.OP(p, q, True, NV, NH)
								self.A.append((p, q))
								if (p == self.dpts[a]):
									self.dpts[2] = p
									self.dpts[a] = q
								else:
									self.dpts[2] = q
									self.dpts[a] = p
								self.deg[p] = self.deg[p] + 1
								self.deg[q] = self.deg[q] + 1
								mark += 1
								break

				elif (self.deg[self.dpts[a]] == 3):
					L = self.InA(self.dpts[a], self.A)
					W = {}
					nf = {}
					for p, q in L:
						pls = self.plaqInv(p, q)
						fi = self.NfPQ(pls, self.A)
						A1 = self.A.copy()
						A1.remove((p, q))
						ff = self.NfPQ(pls, A1)
						Nfp_f = Nfp_i - fi + ff
						W[(p, q)] = np.exp(-self.V * (ff)/self.T)
						nf[(p,q)] = Nfp_f
					Z = np.sum(list(W.values()))
					minn = min(list(W.values()))
					den = Z - minn
					Pr = {}
					n = random.random()
					for i,j in W.items():
						if (i == (self.dpts[a], self.dpts[2]) or i == (self.dpts[2], self.dpts[a])) and mark!=1:
							Pr[i] = (j - minn) / den
						elif(i == (self.dpts[a], self.dpts[2]) or i == (self.dpts[2], self.dpts[a])) and mark==1:
							Pr[i] = 0
						else:
							Pr[i] = j / den
						if(n<Pr[i]):
							p,q = i #max(Pr, key=Pr.get)
							Nfp_i = nf[(p,q)]
							NV, NH = self.OP(p, q, False, NV, NH)
							self.A.remove((p, q))
							if (p == self.dpts[a]):
								self.dpts[2] = p
								self.dpts[a] = q
							else:
								self.dpts[2] = q
								self.dpts[a] = p
							self.deg[p] = self.deg[p] - 1
							self.deg[q] = self.deg[q] - 1
							break
			CF = CF + 1
		return Fpls,nfp,D,Mr

FPLs = pickle.load(open('../I=4/Worm/FPLsConfigU2_worm_1Mruns_100.txt','rb'))
FPL = pickle.load(open('../I=4/ET/FPL_d0+d1+d2_3M.txt','rb'))
pos = pickle.load(open("../I=4/ET/CO_NodesPositions_ET_I=4.txt","rb"))
PL = pickle.load(open("../I=4/ET/PlaquetteCycles1_ET_I=4.txt", "rb"))
HC = pickle.load(open('../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))

# FPL = pickle.load(open('../I=2/FPL_d0+d1_2M.txt','rb'))
# pos = pickle.load(open("../I=4/VC/CO_NodesPositions_VC_I=4.txt","rb"))
# PL = pickle.load(open("../I=4/VC/PlaquetteCycles1_VC_I=4.txt", "rb"))
# HC = pickle.load(open('../I=2/Hcycle_d0+d1flips.txt','rb'))

# FPL = pickle.load(open('../../Square/FPL_maxFP_SQ64.txt','rb'))
# pos = pickle.load(open("../../Square/SQ64_positions.txt","rb"))
# PL_HC = pickle.load(open("../../Square/PlaquetteCycles1_SQ64_periodic.txt", "rb"))
# g = pickle.load(open('../../Square/SQ64.txt','rb'))
# print(len(PL_HC))
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
# print(E)

# e = []
# for i in range(1,65):
# 	e.append((64*i-1,64*i-64))
# 	e.append((i-1,i+4031))
# print(len(e),len(E),len(FPL))

A = FPL.copy()
deg = dict((u,2) for u in HC_N)
g = nx.Graph()
g.add_edges_from(E)
# nx.draw(g,pos,node_size=0)
# plt.show()

x = random.choice(A)[0]
dpts = [x, x, -1]
NR = 100000
obj = EWorm(A,deg,dpts,T = 4,V = -1)
# obj.plotAB(0,0,0)
Fpls,_,_,_ = obj.worm()
# obj.plotAB(0,0,0)
print(len(Fpls))
pickle.dump(Fpls,open('../I=4/Worm/FPLsConfig_worm_T=4_V=-1.txt','wb'))

# plt.plot(list(range(len(nfp)))[::100], nfp[::100],label='NfP,V=-1,T=1')
# plt.scatter(list(range(len(nfp)))[::100], nfp[::100],s=5)
# plt.legend()
# plt.show()
# plt.plot(list(range(len(D)))[::100], D[::100],label='D,V=-1,T=1')
# plt.scatter(list(range(len(D)))[::100], D[::100],s=5)
# plt.legend()
# plt.show()
dfgfd

# for i in Fpls[-100:]:
# 	g = nx.Graph()
# 	g.add_edges_from(i)
# 	pylab.ion()
# 	plt.clf()
# 	nx.draw(g,pos=pos,node_size=5)
# 	pause(0.2)
# 	pylab.show()
# 	plt.show()

Dt = []
T = np.concatenate((np.linspace(0.1,1.15,7),np.linspace(1.2,1.5,5),np.linspace(1.6,2.5,5)),axis=0)   #[1.538,2.222]
MR = []
Nfpt = []
nr = np.concatenate(([80000]*7,[80000]*5,[80000]*5),axis=0)
for k in range(len(T)):
	print(T[k])
	NR = nr[k]
	A = FPL.copy()
	deg = dict((u, 2) for u in g.nodes())
	# g = nx.Graph()
	# g.add_edges_from(E)
	x = random.choice(A)[0]
	dpts = [x, x, -1]
	obj = EWorm(A, deg, dpts, T=T[k], V=-1)
	_,Nfp,D,Mr = obj.worm()
	Dt.append(np.mean(D[-10000:]))
	Nfpt.append(np.mean(Nfp[-10000:]))
	MR.append(Mr)
	print(Dt)
	plt.plot(list(range(len(Nfp)))[::100], Nfp[::100],label='NfP,V=-1,T=%0.2f'%T[k])
	plt.scatter(list(range(len(Nfp)))[::100], Nfp[::100],s=5)
	plt.legend()
	plt.show()
	plt.plot(list(range(len(D)))[::100], D[::100], label='D,V=-1,T=%0.2f'%T[k])
	plt.scatter(list(range(len(D)))[::100], D[::100], s=5)
	plt.legend()
	plt.show()

plt.plot(T,Nfpt,label='V=-1')
plt.scatter(T,Nfpt,s=5)
plt.legend()
plt.show()
import pickle
import matplotlib.pylab as plt
L = pickle.load(open('../../Square/Worm/OP_DvsT_SQ16_2.txt','rb'))
L1 = pickle.load(open('../../Square/Worm/OP_DvsT_SQ64.txt','rb'))
T = L[0]
Dt = L[1]
Dt1 = L1[1]
print(T)
plt.plot(T,Dt,label='V=-1, SQ16')
plt.scatter(T,Dt,s=5)
# plt.plot(T,Dt1,label='V=-1, SQ64')
# plt.scatter(T,Dt1,s=5)
plt.legend()
plt.ylim(-0.2,2.7)
plt.ylim(0,1.1)
plt.show()

fgd

pickle.dump([T,Dt], open('../../Square/Worm/OP_DvsT_SQ64_2.txt','wb'))
#pickle.dump(MR, open('../../Square/Worm/M(r)vsT_SQ64.txt','wb'))
sfds

# pickle.dump([T,Dt], open('../../Square/Worm/OP_DvsT_SQ64_200k.txt','wb'))
# pickle.dump(MR, open('../../Square/Worm/M(r)vsT_SQ64_200k.txt','wb'))
#

import collections as cl
import matplotlib.pylab as plt
# from scipy.optimize import curve_fit
import pickle
# import numpy as np
# Ds = pickle.load(open('../../Square/Worm/OP_Ds_SQ64_500k.txt','rb'))
# L = pickle.load(open('../I=2/Worm/OP_NFPvsT.txt','rb'))
# MR = pickle.load(open('../I=2/Worm/M(r)vsT.txt','rb'))
# mr = pickle.load(open('../I=2/Worm/M(r)_2Mruns_NR.txt','rb'))
# L = pickle.load(open('../../Square/Worm/OP_DvsT_SQ64_200k.txt','rb'))
MR = pickle.load(open('../../Square/Worm/M(r)vsT_SQ16.txt','rb'))
T = [1.538,2.222]
# mr = pickle.load(open('../../Square/Worm/M(r)_SQ64_1Mruns_NR.txt','rb'))
# T = L[0]
# def f(x,B,C,g):
# 	return B/(x**g)+C
# g = []
# beta = []
for k in range(len(T)):   #4,len(T)-2,4):
	m = cl.Counter(MR[k])
	m = {u: v / 10000 for u, v in m.items()}
	m = dict(sorted(m.items()))
	#del m[0.0]
	#popt, pcov = curve_fit(f, np.log(list(m.keys()))[1:4],np.log(list(m.values()))[1:4])
	#print(popt[2])
	#g.append(popt[2])
	#beta.append(1/k)
	plt.scatter(m.keys(), m.values(), s=10)
	plt.plot(m.keys(), m.values(),label='T=%0.02f'%T[k])
	# plt.plot(list(m.keys())[1:4],np.exp(f(np.log(list(m.keys()))[1:4],popt[0],popt[1],popt[2])),'orange')
	# plt.scatter(list(range(len(Ds[k]))), Ds[k], s=5)
	# plt.plot(list(range(len(Ds[k]))), Ds[k], label='T=%0.02f' % T[k])
	#plt.legend()
	#plt.show()
# m = cl.Counter(mr)
# m = {u: v/10000 for u, v in m.items()}
# m = dict(sorted(m.items()))
# del m[0.0]
# popt, pcov = curve_fit(f, np.log(list(m.keys()))[1:4],np.log(list(m.values()))[1:4])
# print(popt[2])
# g.append(popt[2])
# beta.append(0.0)
# plt.scatter(m.keys(), m.values(), s=10)
# plt.plot(m.keys(), m.values(),label='T=inf')
# plt.plot(list(m.keys())[1:4],np.exp(f(np.log(list(m.keys()))[1:4],popt[0],popt[1],popt[2])),'orange')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
# plt.clf()
# plt.scatter(beta,g,s=10)
# plt.show()








'''
					n = random.random()   #ask
						if (i==(self.dpts[a],self.dpts[2]) or i==(self.dpts[2],self.dpts[a])) and mark!=1:
							prob = (j-min(list(W.values())))/(Z - min(list(W.values())))
						elif (i==(self.dpts[a],self.dpts[2]) or i==(self.dpts[2],self.dpts[a])) and mark==1:
							prob = 0
						else:
							prob = j/(Z - min(list(W.values())))
						if(n<prob):
							Nfp_i = nf[i]
							p,q = i
							NV, NH = self.OP(p, q, False, NV, NH)
							self.A.remove((p,q))
							if (p == self.dpts[a]):
								self.dpts[2] = p
								self.dpts[a] = q
							else:
								self.dpts[2] = q
								self.dpts[a] = p
							self.deg[p] = self.deg[p] - 1
							self.deg[q] = self.deg[q] - 1
							break

				elif (self.deg[self.dpts[a]] == 1):
					L = self.NotInA(self.dpts[a], self.A)
					for p, q in L:
						if (p == self.dpts[a] and self.deg[q] == 3):
							L.remove((p, q))
						elif (q == self.dpts[a] and self.deg[p] == 3):
							L.remove((p, q))
					l=0
					for p, q in L:
						if ((p == self.dpts[a] and q == self.dpts[0]) or (p == self.dpts[0] and q == self.dpts[a])) and mark != 0:
							pls = self.plaqInv(p, q)
							fi = self.NfPQ(pls, self.A)
							A1 = self.A.copy()
							A1.append((p, q))
							ff = self.NfPQ(pls, A1)
							Nfp_f = Nfp_i - fi + ff
							if self.accept(Nfp_i, Nfp_f):
								Nfp_i = Nfp_f
								self.dpts[2] = self.dpts[a]
								self.dpts[1] = self.dpts[0]
								self.A.append((p, q))
								self.deg[p] = self.deg[p] + 1
								self.deg[q] = self.deg[q] + 1
								NV, NH = self.OP(p, q, True, NV, NH)
							else:
								pls = self.plaqInv(self.dpts[a],self.dpts[2])
								fi = self.NfPQ(pls, self.A)
								self.A.append((self.dpts[a],self.dpts[2]))
								ff = self.NfPQ(pls, self.A)
								Nfp_i = Nfp_i - fi + ff
								self.deg[self.dpts[2]] = self.deg[self.dpts[2]] + 1
								self.deg[self.dpts[a]] = self.deg[self.dpts[a]] + 1
								self.dpts = [self.dpts[0], self.dpts[2], self.dpts[a]]
								NV, NH = self.OP(self.dpts[a], self.dpts[2], True, NV, NH)
							l=1
							break
					if l==0:
						W = {}
						nf = {}
						for p, q in L:
							pls = self.plaqInv(p, q)
							fi = self.NfPQ(pls, self.A)
							A1 = self.A.copy()
							A1.append((p, q))
							ff = self.NfPQ(pls, A1)
							Nfp_f = Nfp_i - fi + ff
							W[(p, q)] = np.exp(-self.V * (ff) / self.T)   #Nfp_f
							nf[(p, q)] = Nfp_f
						Z = np.sum(list(W.values()))
						for i, j in W.items():
							n = random.random()
							if (i == (self.dpts[a], self.dpts[2]) or i == (self.dpts[2], self.dpts[a])) and mark!=0:
								prob = (j - min(list(W.values()))) / (Z - min(list(W.values())))
							elif (i == (self.dpts[a], self.dpts[2]) or i == (self.dpts[2], self.dpts[a])) and mark==0:
								prob = 0
							else:
								prob = j / (Z - min(list(W.values())))
							if (n < prob):
								Nfp_i = nf[i]
								p,q = i
								NV, NH = self.OP(p, q, True, NV, NH)
								self.A.append((p, q))
								if (p == self.dpts[a]):
									self.dpts[2] = p
									self.dpts[a] = q
								else:
									self.dpts[2] = q
									self.dpts[a] = p
								self.deg[p] = self.deg[p] + 1
								self.deg[q] = self.deg[q] + 1
								mark += 1
								break
'''










