import networkx as nx     
import matplotlib.pylab as plt
import numpy as np
import random
from collections import Counter
import copy
import time
import pickle

# FPLs = pickle.load(open('../I=2/Worm/FPLsConfig_T=0.33_100k.txt','rb'))
gr = pickle.load(open('../I=4/VC/ContractedTilling_VC_I=4.gpickle','rb'))
pos = pickle.load(open('../I=4/VC/CO_NodesPositions_VC_I=4.txt', 'rb'))
FPLs = pickle.load(open('../I=2/Worm/FPLsConfig_T=0.10_V=-1.txt','rb'))
print(len(FPLs))
# pos = pickle.load(open('../I=6/CO_NodesPositions_ET_I=6.txt', 'rb'))
# FPLs = pickle.load(open('../I=6/Worm/FPLsConfigU3_worm_+7(3kruns_5.txt','rb'))

#nx.draw(gr,pos=pos,node_size=0,edge_color = 'grey', width = 1)
# for I in FPLs:
# 	g = nx.Graph()
# 	g.add_edges_from(I)
# 	nx.draw(g,pos=pos,node_size=0, edge_color = 'purple', width = 1, arrows=True) #, connectionstyle='arc3,rad=0.2')
# 	plt.show()
# fhfgj

#pos=pickle.load(open("../../HatTile/Spectre/Delta/Pos_output-6_fkt.pickle", "rb"))
#FPLs=pickle.load(open("../../HatTile/Spectre/Delta/RedLoops_output-6_fkt.txt", "rb"))
# FPLs = pickle.load(open('../../Square/Worm/FPLsConfig_SQ64_10kruns_100_T=-0.4_.txt','rb'))
# print(len(FPLs))
# pos = pickle.load(open("../../Square/SQ64_positions.txt","rb"))

#Flips=pickle.load(open("../I=4/ET/FPLsFlips_300kon3M_d0+d1+d2.txt","rb"))
#FPL = pickle.load(open('../I=4/ET/Hcycle_ET_I=4_d0+d1+d2flips.txt','rb'))
#FPL = pickle.load(open('../I=4/ET/FPL_d0+d1+d2_3M.txt','rb'))
#pos = pickle.load(open('../I=4/ET/CO_NodesPositions_ET_I=4.txt', 'rb'))

'''
#FPL=FPLs[-1]
X=BigP[2]
ED_flip=[]
for i in range(len(X)):
	if(i%2!=0):
		for u,v in X[i]:
			ED_flip.append((v,u))
	else:
		ED_flip.extend(X[i])

St = time.time()
q=0
for j in FPLs[200:201]: 
	loop=[]
	ED_flip=j.copy()
	while(ED_flip!=[]):
		IV=ED_flip[0][0]
		p=ED_flip[0][1]
		L = [ED_flip[0][0]]
		ED_flip.remove(ED_flip[0])
		while(p!=IV):
			c=0
			for i in ED_flip:
				if(p==i[0]):
					c=c+1
					p=i[1]
					L.append(i[0])
					ED_flip.remove(i)
					break
				elif(p==i[1]):
					c=c+1
					p=i[0]
					L.append(i[1])           #,i[0]))  #i
					ED_flip.remove(i)
					break
		
		if(c==0):
			print('*********')
		loop.append(L)
		#if(len(L)>14):
		#	print(L)
		#	pickle.dump(L, open("../../HatTile/Spectre/Delta/BoundaryLoop_output-4.txt", "wb"))
		#	break
		L=[]
	
	#print(q, len(loop))
	#pickle.dump(loop, open("../../Square/Worm/Loops/Loops_FPLs%i.txt"%q, "wb"))    
	q=q+1

print(time.time()-St)

###############################################

St = time.time()
q=0
for kk in Flips[:1]:
	print(q)
	FPL.remove(kk[0])
	FPL.remove(kk[1])
	FPL.append(kk[2])
	FPL.append(kk[3])
	
	if(q%50==0): 
		#p=FPL
		loop=[]
		ED_flip=FPL.copy()
		while(ED_flip!=[]):
			IV = ED_flip[0][0]
			p = ED_flip[0][1]
			L = [ED_flip[0][0]]
			ED_flip.remove(ED_flip[0])
			while(p!=IV):
				c=0
				for i in ED_flip:
					if(p==i[0]):
						c=c+1
						p=i[1]
						L.append(i[0])
						ED_flip.remove(i)
						break
					elif(p==i[1]):
						c=c+1
						p=i[0]
						L.append(i[1])  #,i[0]))  #i
						ED_flip.remove(i)
						break
			if(c==0):
				print('*****')
			loop.append(L)
			L=[]

		#print(len(loop))
		#pickle.dump(loop, open("../I=4/Worm/Test_WormvsFlip/FPLsConfig_flip/Loops_FPLs%s.txt"%(int(q/50)), "wb"))    
		#print(loop[-1])
	q=q+1

print(time.time() - St)
#pickle.dump(FPL, open('../AB_HNew/FPL_d0+d1_2M.txt', 'wb'))   
'''
######################################################################################

St = time.time()
q=0
LLd = []
for j in FPLs:  #[:]:
	g=nx.Graph()
	g.add_edges_from(j)
	loop=[]
	gc = g.copy()
	LL = []
	while(list(g.nodes()) != []):
		IV = random.choice(list(g.nodes()))
		p = list(g.neighbors(IV))[0]
		L = [IV,p]
		flag = True
		while flag:
			nn = list(g.neighbors(p))
			c = 0
			for I in nn:
				if(I not in L):
					c = 1
					L.append(I)
					p=I
					break
			if c == 0:
				flag = False
		loop.append(L)
		LL.append(len(L))
		#print(len(L))
		#if(len(L)>14):
		#	print(L)
		#	pickle.dump(L, open("../../HatTile/Spectre/Delta/BoundaryLoop_output-6_fkt.txt", "wb"))
		#	break
		g.remove_nodes_from(L)
		L=[]

	print(q, len(loop))
	#pickle.dump(loop, open("../I=4/Worm/Loops_New/Loops_FPLs%i_T=8.00_V=-1.txt"%q, "wb"))
	q=q+1
	LLd.append(dict(Counter(LL)))

D = {}
for d in LLd:
	for u,v in d.items():
		if u in D.keys():
			D[u].append(v)
		else:
			D[u] = [v]
D = {u:np.mean(v) for u,v in D.items()}
D = dict(sorted(D.items()))
print(D)

HD = {4: 4.790983606557377, 6: 2.9623655913978495, 8: 1.8617021276595744, 10: 3.94672131147541, 12: 2.3597122302158273, 14: 1.8605769230769231, 16: 2.889952153110048, 18: 2.909547738693467, 20: 1.032258064516129, 22: 1.1647058823529413, 24: 1.2435897435897436, 26: 1.0689655172413792, 28: 1.0, 30: 1.05, 32: 1.0, 34: 1.0555555555555556, 36: 1.0, 38: 1.0, 40: 2.9130434782608696, 42: 1.0, 46: 1.0, 48: 1.0, 50: 1.0, 52: 1.0, 54: 1.0, 56: 1.0, 60: 1.0, 64: 1.0, 66: 1.0, 68: 1.0, 70: 1.0, 72: 1.0, 74: 1.0, 76: 1.2, 78: 1.0, 80: 1.0, 82: 1.0, 88: 1.0, 90: 1.0, 92: 1.0, 96: 1.0, 106: 1.0, 108: 1.0, 112: 1.0, 114: 1.0, 118: 1.0, 126: 1.0, 130: 1.0, 132: 1.0, 134: 1.0, 136: 1.0, 140: 1.0, 142: 1.0, 148: 1.0, 150: 1.0, 152: 1.0, 154: 1.0, 158: 1.0, 162: 1.0, 164: 1.0, 168: 1.0, 172: 1.0, 174: 1.0, 176: 1.0, 178: 1.0, 180: 1.0, 184: 1.0, 186: 1.0, 190: 1.0, 194: 1.0, 198: 1.0, 200: 1.0, 202: 1.0, 204: 1.0, 206: 1.0, 208: 1.0, 212: 1.0, 214: 1.0, 216: 1.0, 218: 1.0, 220: 1.0, 222: 1.0, 224: 1.0, 226: 1.0, 228: 1.0, 230: 1.0, 232: 1.0, 234: 1.0, 238: 1.0, 240: 1.0, 242: 1.0, 244: 1.0, 246: 1.0, 248: 1.0, 250: 1.0, 252: 1.0, 254: 1.0, 256: 1.0, 258: 1.0, 260: 1.0, 262: 1.0, 264: 1.0, 266: 1.0, 268: 1.0, 278: 1.0, 280: 1.0, 314: 1.0, 316: 1.0, 328: 1.0}
LD8 = {4: 6.72, 6: 3.8181818181818183, 8: 2.4347826086956523, 10: 2.76, 12: 1.4, 14: 2.1052631578947367, 16: 2.4, 18: 2.272727272727273, 20: 1.0, 22: 1.0, 24: 1.0, 28: 1.0, 32: 1.0, 34: 1.0, 36: 1.0, 46: 1.0, 50: 1.0, 198: 1.0, 208: 1.0, 210: 1.0, 216: 1.0, 218: 1.0, 220: 1.0, 224: 1.0, 230: 1.0, 234: 1.0, 236: 1.0, 238: 1.0, 240: 1.0, 246: 1.0, 252: 1.0, 266: 1.0, 280: 1.0}
LD5 = {4: 5.7, 6: 2.1, 8: 3.8, 10: 1.5, 12: 1.8571428571428572, 14: 2.6, 16: 1.8, 18: 1.75, 20: 1.0, 22: 1.0, 24: 1.6666666666666667, 26: 1.0, 28: 1.0, 38: 1.0, 44: 1.0, 104: 1.0, 246: 1.0, 250: 1.0, 264: 1.0, 274: 1.0, 280: 1.0}
#LD3 = {4: 4.6, 6: 2.75, 8: 3.0, 10: 2.2, 12: 2.4, 14: 1.6666666666666667, 16: 1.8, 18: 1.5, 20: 1.2, 48: 1.0, 60: 1.0, 64: 1.0, 214: 1.0, 226: 1.0, 244: 1.0, 280: 1.0, 288: 1.0}
#LD1 = {4: 7.181818181818182, 6: 2.272727272727273, 8: 1.8181818181818181, 10: 3.272727272727273, 12: 2.2, 14: 1.5454545454545454, 16: 3.142857142857143, 18: 1.8333333333333333, 20: 1.0, 24: 1.0, 26: 1.5, 28: 1.0, 30: 1.0, 144: 1.0, 166: 1.0, 264: 1.0, 280: 1.0, 282: 1.0, 284: 1.0, 298: 1.0}

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(1, 1, figsize=(71,78))
plt.scatter(HD.keys(),HD.values(),s=7000,label='V = -1, T = %.1f'%1000)
plt.scatter(LD5.keys(),LD5.values(),s=7000,label='V = -1, T = %.1f'%0.50)

plt.ylabel(r'Mean Frequency', fontsize = 300)   #$-log(\langle |h(p)|^{2} \rangle)$
plt.xlabel(r'Loop Lengths',fontsize = 300)
plt.tick_params(axis='x', labelsize=300, length=80, width=20)
plt.tick_params(axis='y', labelsize=300, length=80, width=20)
plt.legend(fontsize = 250, loc='upper right')
plt.savefig('FreqvsLL_.pdf')
# plt.xlim(0,100)
# plt.show()
Seeds

#########################################################
'''
St = time.time()
q=0
for kk in Flips[:1]:
	print(q)
	FPL.remove(kk[0])
	FPL.remove(kk[1])
	FPL.append(kk[2])
	FPL.append(kk[3])
	
	if(q%50==0): 
		g=nx.Graph()
		g.add_edges_from(FPL)
		loop=[]
		gc = g.copy()
		while(list(g.nodes()) != []):
			IV = random.choice(list(g.nodes()))
			p = list(g.neighbors(IV))[0]     
			L = [IV,p]
			flag = True
			while flag:
				nn = list(g.neighbors(p)) 
				c = 0
				for I in nn:
					if(I not in L):
						c = 1
						L.append(I)
						p=I
						break
				if c == 0:
					flag = False
			loop.append(L)
			g.remove_nodes_from(L)
			L=[]
	
		#print(q, len(loop))
		#pickle.dump(loop, open("../../Square/Worm/Loops/Loops_FPLs%i.txt"%q, "wb"))    
	q=q+1

print(time.time() - St)
print(len(loop))
'''
g = nx.Graph()
for u in loop:
	e=[]
	for I in range(len(u)-1):
		e.append((u[I],u[I+1]))
	e.append((u[-1],u[0]))
	
	g.add_edges_from(e)
nx.draw(g,pos=pos,node_size=2)
plt.show()


















'''
FPLs_U2

0-50000 / 50 = 2000-3000   xx
1000-11000 / 50 = 3000-3200
0-1000 / 10 = 3200-3300

0-10000 / 10 = 2000-3000
'''


