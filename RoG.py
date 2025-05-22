import networkx as nx     
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import collections as CL
import numpy as np
import pickle

G = pickle.load(open("../I=4/VC/ContractedTilling_VC_I=4.gpickle","rb"))
G1 = pickle.load(open("../I=4/ET/ContractedTilling_ET_I=4.gpickle","rb"))
# G2 = pickle.load(open("../I=6/ContractedTilling_ET_I=6.gpickle","rb"))
pos = pickle.load(open("../I=4/VC/CO_NodesPositions_VC_I=4.txt","rb"))
pos1 = pickle.load(open("../I=4/ET/CO_NodesPositions_ET_I=4.txt","rb"))
# pos2 = pickle.load(open("../I=6/CO_NodesPositions_ET_I=6.txt","rb"))
# pos = pickle.load(open("../../Square/SQ64_positions.txt","rb"))

def RSD(L,pos):
	max = 0
	s=0
	for i in L:
		s=s+pos[i]	
	rc = s/len(L)

	for i in L:
		r = np.linalg.norm(pos[i]-rc)
		if(r >= max):
			max = r
	return max
	
	'''
	g = nx.Graph()
	e=[]
	for i in range(len(L)-1):
		e.append((L[i],L[i+1]))
	g.add_edges_from(e)
	circle1 = plt.Circle(rc, max, color='r')
	plt.gca().add_patch(circle1)
	nx.draw(g,pos=pos,node_size=5)
	plt.show()
	'''

'''
def RoG(L,pos):
	s=0
	for i in L:
		s=s+pos[i]
		
	rc = s/len(L)
	s=0
	for i in L:
		s=s + np.linalg.norm(pos[i]-rc)**2
	rg2 = s/len(L)
	return rg2
'''

################################ R vs s ##################################

def R(name, nn, pos, N, L):
	RG2={}
	for i in range(0,N):
		print(i)
		Loops = pickle.load(open('%s/Loops_%s%i.txt'%(name, nn, i),'rb'))
		for j in Loops:
			flag=False
			r = RSD(j,pos)
			for k in RG2.keys():
				if(r > k-0.1 and r < k+0.1):
					p=k
					flag = True
					break
			if flag:
				RG2[p].append(len(j))
			else:
				RG2[r] = [len(j)]

	RG={}
	for u,v in RG2.items():
		RG[u]= np.mean(v)

	RG = dict(sorted(RG.items()))
	#x = sorted(list(RG.keys()))
	Y=[]
	X=[]
	Zy=[]
	Zx=[]
	for i in list(RG.keys()):      #x:
		Zy.append(i**(-1.35) * RG[i])  #-1.568
		Zx.append(i*(1/L))
		Y.append(np.log(RG[i]))
		X.append(np.log(i))
	return RG,X,Y,Zx,Zy

#RG,X,Y,_,_ = R('../../Square/Loops_SQ64','FPLs',pos,10000,64)
#RG,X,Y,Zx,Zy = R('../I=2/Worm/Loops','FPLs',pos, 4000, 26)
#RG1,X1,Y1,Zx1,Zy1 = R('../I=4/Worm/Loops','FPLs',pos1, 100000, 152)
#RG2,X2,Y2,Zx2,Zy2 = R('../DDLs/DDLoopsU1_relabel','DDLs',pos, 4000,26)
#RG3,X3,Y3,Zx3,Zy3 = R('../DDLs/DDLoopsU2_relabel','DDLs',pos1, 20000,152)

#pickle.dump([RG,X,Y,Zx,Zy],open('../I=2/s(R)vsR_4000loops.txt','wb'))
#pickle.dump([RG1,X1,Y1,Zx1,Zy1],open('../I=4/Worm/s(R)vsR_U2_100kloops.txt','wb'))

RG2,X2,Y2,Zx2,Zy2 = pickle.load(open('../I=2/Worm/s(R)vsR_U1_10000loops.txt','rb'))
RG3,X3,Y3,Zx3,Zy3 = pickle.load(open('../I=4/Worm/s(R)vsR_U2_100kloops.txt','rb'))

def f(x, A, B):
    return A*x + B

s2,inter2,_,_,se2 = linregress(X2, Y2)          # SQ64-[:200]         
print(s2,inter2,se2)
s3,inter3,_,_,se3 = linregress(X3, Y3)            
print(s3,inter3,se3)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax1 = plt.subplots(figsize=(64, 60))

#ax1.scatter(list(RG1.keys()), list(RG1.values()), s=5, c='orange', label = 'FPL_U2 {8_4}')
#ax1.scatter(list(RG.keys()), list(RG.values()), s=5, c='g', label = 'FPL_U1 {8_2}')
ax1.scatter(list(RG3.keys()), list(RG3.values()), s=3000, label = '$U_2$ AB region, 14992 Nodes')
ax1.scatter(list(RG2.keys()), list(RG2.values()), s=3000, label = '$U_1$ AB region, 464 Nodes')

#ax1.plot(np.exp(np.array(X1)), np.exp(f(np.array(X1), s1, inter1)), 'orange')  
#ax1.plot(np.exp(np.array(X)), np.exp(f(np.array(X), s, inter)), 'g') 
ax1.plot(np.exp(np.array(X3)), np.exp(f(np.array(X3), s3, inter3)),'gray',linewidth = 30, label = 'Linear_Fit')
# ax1.plot(np.exp(np.array(X2)), np.exp(f(np.array(X2), s2, inter2)),'gray',linewidth = 20)
 
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_ylabel(r'$s$')   #Average Loop Size (s) [log scale]')
# ax1.set_xlabel(r'$R$')   #Radius (R) [log scale]')
# ax1.set_xlim([-1,10**2])
# ax1.set_ylim([-1,10**4])
# ax1.legend()

plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$s$', fontsize = 250)
plt.xlabel(r'$R$', fontsize = 220)
plt.tick_params(axis='x', labelsize=175, length=60, width=10)
plt.tick_params(axis='y', labelsize=180, length=60, width=10)
plt.legend(fontsize = 170, loc='upper left')

ax2 = fig.add_axes([0.54, 0.18, 0.35, 0.29])
ax2.scatter(Zx3, Zy3, s=1000)
ax2.plot(Zx3, Zy3, linewidth = 10)
ax2.scatter(Zx2, Zy2, s=1000)
ax2.plot(Zx2, Zy2, linewidth = 10)
#ax2.scatter(Zx1, Zy1, s=2, c='orange')
#ax2.plot(Zx1, Zy1, c='orange')
#ax2.scatter(Zx, Zy, s=2, c='g')
#ax2.plot(Zx, Zy, c='g')
ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.set_title('$d_{f} = 1.568$', fontsize = 150)
ax2.set_ylabel('$R^{-d_{f}} s(R)$',fontsize = 150)
ax2.set_xlabel('$RL^{-1}$',fontsize = 150)
ax2.tick_params(axis='x', labelsize=100, length=60, width=10)
ax2.tick_params(axis='y', labelsize=100, length=60, width=10)

plt.savefig('Ps_AB.pdf')
plt.show()
sdfd

############################ n(R) vs R #################################
'''
def nR(name,nn,pos,A,N,L):
	RG={}
	for i in range(N):
		print(i)
		Loops = pickle.load(open('%s/Loops_%s%i.txt'%(name, nn,i),'rb'))
		for j in Loops:
			flag = False
			r = RSD(j,pos)        #np.sqrt(RoG(j,pos))
			for k in RG.keys():
				if(r > k-0.2 and r < k+0.2):
					p=k
					flag = True
					break
			if flag:
				RG[p] = RG[p]+1
			else:
				RG[r] = 1

	RG = dict(sorted(RG.items()))
	Zx=[]
	Zy=[]
	for u,v in RG.items():
		RG[u] = v/(np.pi * (A**2)* N) 
		Zx.append(u*(1/L))
		Zy.append(u**3*RG[u])
	return RG,Zx,Zy

#RG = nR('../AB Tiling New/Square/Loops_SQ64','FPLs',pos,1)
#RG,Zx,Zy = nR('../I=2/FPLsConfig_d0+d1','FPLs',pos, 13, 4000, 26)
#RG1,Zx1,Zy1 = nR('../I=4/Worm/Loops','FPLs',pos1, 76, 60000, 152)
#RG2,Zx2,Zy2 = nR('../DDLs/DDLoopsU1_relabel','DDLs',pos,13,4000,26)
#RG3,Zx3,Zy3 = nR('../DDLs/DDLoopsU2_relabel','DDLs',pos1,76,10000,152)

#pickle.dump([RG,Zx,Zy],open('../I=2/n(R)vsR_4000loops.txt','wb'))
#pickle.dump([RG1,Zx1,Zy1],open('../I=4/Worm/n(R)vsR_60000loops.txt','wb'))

RG2,Zx2,Zy2 = pickle.load(open('../I=2/Worm/n(R)vsR_U1_10000loops.txt','rb'))
RG3,Zx3,Zy3 = pickle.load(open('../I=4/Worm/n(R)vsR_U2_60000loops.txt','rb'))

fig, ax1 = plt.subplots()

#ax1.scatter(list(RG1.keys()),list(RG1.values()),c='orange',s=5,label='FPL_U2 {8_4}')
#ax1.scatter(list(RG.keys()),list(RG.values()),c='g',s=5,label='FPL_U1 {8_2}')
ax1.scatter(list(RG3.keys()),list(RG3.values()),c='orange',s=5,label='FPL_U2 {8_4}')
ax1.plot(list(RG3.keys()),list(RG3.values()),c='orange')
ax1.scatter(list(RG2.keys()),list(RG2.values()),c='g',s=5,label='FPL_U1 {8_2}')
ax1.plot(list(RG2.keys()),list(RG2.values()),c='g')

ax1.set_xlabel('Radius (R)')
ax1.set_ylabel('Average number density of loops n(R)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim([10**(-2),10**4])
ax1.set_ylim([10**(-12),0])
ax1.legend()

ax2 = fig.add_axes([0.22, 0.2, 0.35, 0.25])
ax2.scatter(Zx3, Zy3, s=2, c='orange')
ax2.plot(Zx3, Zy3, c='orange')
ax2.scatter(Zx2, Zy2, s=2, c='g')
ax2.plot(Zx2, Zy2, c='g')
#ax2.scatter(Zx1, Zy1, s=2, c='orange')
#ax2.plot(Zx1, Zy1, c='orange')
#ax2.scatter(Zx, Zy, s=2, c='g')
#ax2.plot(Zx, Zy, c='g')
ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.set_ylabel('$R^{3} n(R)$',fontsize = 7)
ax2.set_xlabel('$RL^{-1}$',fontsize = 7)

plt.show()
Dxfs
'''
############################ P(s) vs s #################################
def angle(b,c,a,ps):
	x = ps[b]-ps[c]
	y = ps[a]-ps[c]
	ang = np.arccos(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
	#print(ang)
	if(np.round(ang,2) == np.round(np.pi,2)):
		#print('*')
		return True
	else:
		return False

def length(j,a):
	if(a==13):
		g = G
		ps = pos
	else:
		g = G1
		ps = pos1
		
	l=0
	for k in range(len(j)-1):
		if(g.degree(j[k]) in [3,4]):
			l = l+1
		else:
			ang = angle(j[k-1],j[k],j[k+1],ps)
			if ang:
				l=l+1000
			else:
				l=l+1
	if(g.degree(j[-1]) in [3,4]):
		l = l+1
	else:
		ang = angle(j[-2],j[-1],j[0],ps)
		if ang:
			l=l+1000
		else:
			l=l+1
	
	return l

def Ps(name,nn,pos,A,N,L):
	PLs={}    #area density of loops of size s
	for i in range(0,N):
		print(i)
		Loops = pickle.load(open('%s/Loops_%s%i.txt'%(name,nn,i),'rb'))
		for j in Loops:
			flag=False
			l=len(j)    #length(j,A)        
			for k in PLs.keys():
				if(l > k-0.1 and l < k+0.1):
					p=k
					flag = True
					break
			if flag:
				PLs[p] = PLs[p]+1
			else:
				PLs[l] = 1
			'''
			l=len(j)
			if (l not in PLs):
				PLs[l] = 1
			else:
				PLs[l] = PLs[l]+1
			'''
	PLs = dict(sorted(PLs.items()))
	Zx=[]
	Zy=[]
	for u,v in PLs.items():
		PLs[u] = v/(np.pi * (A**2) * N)    #np.pi *       
		Zx.append(u*L**(-1.56))
		Zy.append(u**(2.282)*PLs[u])
		 
	return PLs,Zx,Zy

#PLs1,Zx1,Zy1 = Ps('../I=4/Worm/Loops','FPLs',pos1,76,120000,152)
#PLs,Zx,Zy = Ps('../I=2/Worm/Loops','FPLs',pos,13,13000,26)
#PLs2,Zx2,Zy2 = Ps('../I=6/Worm/Loops','FPLs',pos2,442,691,885)
#PLs,Zx,Zy = Ps('../DDLs/DDLoopsU1_relabel','DDLs',pos,13,4000,26)
#PLs1,Zx1,Zy1 = Ps('../DDLs/DDLoopsU2_relabel','DDLs',pos1,76,20000,152)

#pickle.dump([PLs,Zx,Zy],open('../I=4/Worm/Test_WormvsFlip/P(s)vss_U2_25000loops_flip.txt','wb'))
#pickle.dump([PLs1,Zx1,Zy1],open('../I=4/Worm/P(s)vss_U2_120kloops.txt','wb'))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(1, 1, figsize=(68,60), sharex=True, sharey=True)  #(78,78)

L1 = pickle.load(open('../I=4/Worm/P(s)vss_U2_120kloops.txt','rb'))
L = pickle.load(open('../I=2/Worm/P(s)vss_U1_10000loops.txt','rb'))
#L = pickle.load(open('../I=4/Worm/Test_WormvsFlip/P(s)vss_U2_25000loops_flip.txt','rb'))
# PLs = pickle.load(open('../../Square/P(s)vss_SQ64_10000loops.txt','rb'))
#PLs = pickle.load(open('../../Square/Worm/P(s)vss_SQ64_30000loops.txt','rb'))

PLs = L[0]
Zx = L[1]
Zy = L[2]
PLs1 = L1[0]
Zx1 = L1[1]
Zy1 = L1[2]

def f(x,A,b):
	return A*x+b

l1 = np.log(list(PLs.keys()))
l2 = np.log(list(PLs.values()))
l11 = np.log(list(PLs1.keys()))
l21 = np.log(list(PLs1.values()))
s,inter,_,_,se = linregress(l1[8:65], l2[8:65])            #[8:65]-fplWorm,flip  [:150]-SQ64  [:30]-DDL
print(s,inter,se)
s1,inter1,_,_,se1 = linregress(l11[8:65], l21[8:65])   #[8:1000]-fplWorm   [5:30]-fplflip   [:300]-DDL
print(s1,inter1,se1)

# ax1.scatter(list(PLs2.keys()),list(PLs2.values()),s=5,c='b',label='FPL_U3 {8_6}')
# ax1.plot(list(PLs2.keys()),list(PLs2.values()),c='b')
plt.scatter(list(PLs1.keys()),list(PLs1.values()),s=3000,label= 'U2 AB region, 14992 Nodes')
plt.plot(list(PLs1.keys()),list(PLs1.values()),linewidth = 10)
plt.scatter(list(PLs.keys()),list(PLs.values()),s=3000, label='U1 AB region, 464 Nodes')
plt.plot(list(PLs.keys()),list(PLs.values()),linewidth = 10)
plt.plot(np.exp(np.array(l11[8:65])), np.exp(f(np.array(l11[8:65]), s1, inter1)),c = 'gray',linewidth = 30,label='Linear-Fit')
# plt.plot(np.exp(np.array(l1[8:65])), np.exp(f(np.array(l1[8:65]), s, inter)), c='gray', linewidth = 20, label='Linear-Fit')
#ax1.scatter(list(PLs3.keys()),list(PLs3.values()),s=5,c='orange',label='DDL_U2 {8_4}')

plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$P(s)$', fontsize = 210)
plt.xlabel(r'$s$',fontsize = 220)
plt.tick_params(axis='x', labelsize=170, length=60, width=10)
plt.tick_params(axis='y', labelsize=170, length=60, width=10)
plt.legend(fontsize = 160, loc='upper right')

ax2 = fig.add_axes([0.2, 0.18, 0.35, 0.29])
# ax2.scatter(Zx3, Zy3, s=2, c='orange')
# ax2.scatter(Zx2, Zy2, s=2, c='g')
# ax2.plot(Zx2, Zy2, c='b')
ax2.scatter(Zx1[::10], Zy1[::10], s=1000)
ax2.plot(Zx1[::10], Zy1[::10], linewidth=10)
ax2.scatter(Zx, Zy, s=1000)
ax2.plot(Zx, Zy, linewidth=10)
ax2.set_xscale('log')
# ax2.set_yscale('log')
# ax2.set_ylim([0,50])
ax2.set_title(r'$d_{f} = 1.56, \tau = 2.282$', fontsize = 150)
ax2.set_ylabel(r'$s^{\tau} P(s)$',fontsize = 150)
ax2.set_xlabel(r'$s L^{-d_{f}}$',fontsize = 150)
ax2.tick_params(axis='x', labelsize=100, length=60, width=10)
ax2.tick_params(axis='y', labelsize=100, length=60, width=10)

plt.savefig('Ps_AB.pdf')
plt.show()

Dfsf







'''
############################ P(s) vs s #################################

#PL - probability of loop length s (number of loops of length s / number of loops of all lengths)

PL={}    
for i in range(2000):
	Loops = pickle.load(open('FPLsConfig_d0+d1/Loops_FPLs%i.txt'%i,'rb'))
	for j in Loops:
		s=len(j)
		if (l not in PL):
			PL[s] = 1
		else:
			PL[s] = PL[s]+1

print(sorted(PL.keys()))

N = sum(list(PL.values()))
for u,v in PL.items():
	PLs[u] = v/N

plt.scatter(list(PL.keys()),list(PL.values()))

plt.xlabel('Loop Lengths (s)')
plt.ylabel('Probability distribution of loop lengths P(s)')

plt.xscale('log')
plt.yscale('log')

plt.show()	

def Ps(name, name1, pos):
	PLs = pickle.load(open('%s.txt'%(name),'rb')) 
	RG,_,_ = R('%s'%(name1),pos)
	L=[] 
	for i in PLs[:1000000:500]:
		L.extend(i) 
	PLs = CL.Counter(L)  
	for u,v in PLs.items():
		PLs[u] = v/(np.pi * (RG[u]**2))
	return PLs

PLs = Ps('PL_1M_d0+d1_NewAlgo', 'FPLsConfig_d0+d1', pos)
PLs1 = Ps('PL_1.5M_d0+d1+d2_NewAlgo', 'FPLsConfig_d0+d1+d2', pos1)
'''




