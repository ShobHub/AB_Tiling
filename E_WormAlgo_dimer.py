import networkx as nx
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from matplotlib.pyplot import pause
import numpy as np
import pickle
import random
import pylab

class EWorm():
	def __init__(self,M,PL_HC,E,T,V,C):
		self.C = C
		self.M = M
		self.PL_HC = PL_HC
		self.E = E
		g = nx.Graph()
		g.add_edges_from(E)
		self.g = g
		self.T = T
		self.V = V
	def plotAB(self,si,sj,sk):
		EC=[]
		W=[]
		for u,v in self.g.edges():
			if((u,v) in self.M or (v,u) in self.M):
				EC.append('purple')
				W.append(2.5)
			else:
				EC.append('black')
				W.append(0.7)
		NC = []
		w = []
		for u in self.g.nodes():
			if (u==si):
				NC.append('red')
				w.append(30)
			elif (u==sj):
				NC.append('green')
				w.append(30)
			elif (u==sk):
				NC.append('blue')
				w.append(30)
			else:
				NC.append('steelblue')
				w.append(2)
		pylab.ion()
		plt.clf()
		nx.draw(self.g,pos=pos,node_size=w,edge_color=EC,width=W,node_color=NC) #,with_labels=True,font_size=8)
		pause(0.1)
		pylab.show()
		#plt.show()
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
	def plaqInv(self,u,v,e):
		pl = []
		a,b = e
		for i in self.PL_HC:
			if(((u,v) in i or (v,u) in i or (a,b) in i or (b,a) in i) and i not in pl):
				pl.append(i)
		return pl
	def accept(self, NfP_i,NfP_f):
		kB = 1
		prob = min(1, np.exp((self.V * NfP_i - self.V * NfP_f) / (kB * self.T)))
		u = random.uniform(0, 1)
		if (u<prob):
			return True
		else:
			return False
	def NotInM(self, u):
		L = []
		for p, q in self.E:
			if (p == u or q == u):
				if ((p, q) not in self.M and (q, p) not in self.M):
					L.append((p, q))
		return L
	def InM(self, u):
		for p,q in self.M:
			if (p == u):
				return (p,q),q
			elif (q == u):
				return (p,q),p
	def worm(self,N,c):
		NfP_i = self.NfPQ(self.PL_HC,self.M)
		Nfp = []
		Mm = []
		while N>0:
			print(N)
			Nfp.append(NfP_i)
			if c!=0:
				dim = random.choice(self.M)
				si,sj = dim
				sk = -1
				c = self.C
			else:
				c = self.C
			while sj!=si and sk!=si and c>0:
				#print(c,"*")
				#self.plotAB(si, sj, sk)
				empE = self.NotInM(sj)
				for u,v in empE:
					pls = self.plaqInv(u,v,dim)
					fi = self.NfPQ(pls,self.M)
					M1 = self.M.copy()
					M1.remove(dim)
					M1.append((u,v))
					ff = self.NfPQ(pls,M1)
					NfP_f = NfP_i - fi + ff
					if self.accept(NfP_i, NfP_f):
						break
				self.M.remove(dim)
				self.M.append((u,v))
				NfP_i = NfP_f
				if (u == sj):
					sk = v
				else:
					sk = u
				dim,sj = self.InM(sk)
				c -= 1
			N -= 1
			if(N<=50):
				#self.plotAB(si, sj, sk)
				Mm.append(self.M)
		return Nfp,Mm    #self.M

pos = pickle.load(open("../I=4/VC/CO_NodesPositions_VC_I=4.txt","rb"))
PL = pickle.load(open("../I=4/VC/PlaquetteCycles1_VC_I=4.txt", "rb"))
HC = pickle.load(open('../I=2/Hcycle_d0+d1flips.txt','rb'))
HC.extend([(16,1),(15,1),(5468,1),(92,1),(114,1),(112,1),(93,1),(5498,1),(7,37),(37,6),(7,5469),(5469,16),(16,37),(5469,5468),(5468,5467),(5467,92),(92,115),(115,114),(114,113),(113,112),(112,111),(111,93),(93,5497),(5497,5498),(5498,5499),(5499,15),(15,37)])
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
g = nx.Graph()
g.add_edges_from(E)
G = g.copy()
for u,v in g.degree():
	if(v==8):
		nn = list(g.neighbors(u))
		nnn = []
		for i in nn:
			nnn.extend(list(g.neighbors(i)))
		G.remove_nodes_from(nn)
		G.remove_nodes_from(set(nnn))
	#if(np.linalg.norm(pos[u]-(0,0))<=5  and u in G.nodes()):
	#	G.remove_node(u)

E = list(G.edges())
M = nx.max_weight_matching(G)
runs = 10
den = {(u,v):0 for u,v in E}
for k in range(runs):
	print(k)
	#M = nx.max_weight_matching(g)   #pickle.load(open("../DDLs/MaxMatchU1_relabel/MaxMatch_10.txt", "rb"))
	obj = EWorm(list(M),PL_HC, E, T=1, V=5, C=50)
	Nfp,Mm = obj.worm(850,1)
	for u,v in E:
		for m in Mm:
			if((u,v) in m or (v,u) in m):
				den[(u,v)] += 1
	print(Nfp)
	plt.plot(list(range(len(Nfp))), Nfp,label='NfP,V=-1,T=0.3333')
	plt.scatter(list(range(len(Nfp))), Nfp,s=5)
	plt.legend()
	plt.show()

den = {u:(0,0,1,v/(50*runs)) for u,v in den.items()}
EC=[]
for u,v in G.edges():
	if((u,v) in den.keys()):
		EC.append(den[(u,v)])
	elif((v,u) in den.keys()):
		EC.append(den[(v,u)])
nx.draw(G,pos=pos,node_size=0,edge_color=EC,width=2)  #,with_labels=True)
#plt.savefig('../../DMRG/ClassicalMC/Cmc_TV=0.33.pdf')
plt.show()
#pickle.dump(den, open('../../DMRG/ClassicalMC/Density_Cmc_TV=0.33.txt','wb'))










