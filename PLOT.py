import matplotlib.pylab as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import numpy as np
import pickle


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


##### s(R) vs R #######
'''
RG = pickle.load(open('../DDLs/s(R)vsR_DDLsU1_2000loops_relabel.txt','rb'))
RG3 = pickle.load(open('../DDLs/s(R)vsR_DDLsU2_2000loops_relabel.txt','rb'))
L = pickle.load(open('../I=4/ET/s(R)vsR_FPLsU1nU2_1M_2000loops.txt','rb'))
RG4 = pickle.load(open('../../Square/s(R)vsR_SQ64_2000loops.txt','rb'))

RG1 = L[0]
RG2 = L[1]

X,Y = RR(RG)
X1,Y1 = RR(RG1)
X2,Y2 = RR(RG2)
X3,Y3 = RR(RG3)
X4,Y4 = RR(RG4)

popt, pcov = curve_fit(f, X, Y)   
popt1, pcov1 = curve_fit(f, X1, Y1)   
popt2, pcov2 = curve_fit(f, X2, Y2)
popt3, pcov3 = curve_fit(f, X3, Y3)
popt4, pcov4 = curve_fit(f, X4[2:100], Y4[2:100])
   
print(popt, popt1, popt2, popt3, popt4)

plt.scatter(X2, Y2, s=10, label = 'U2 {8_4} - FPL')
plt.scatter(X1, Y1, s=10, label = 'U1 {8_2} - FPL')
#plt.scatter(X3, Y3, s=10, label = 'U2 {8_4} - DDL')
#plt.scatter(X, Y, s=10, label = 'U1 {8_2} - DDL')
#plt.scatter(X4[:100], Y4[:100], s=10, label = 'SQ64')

plt.plot(np.array(X2), f(np.array(X2), popt2[0], popt2[1]))
plt.plot(np.array(X1), f(np.array(X1), popt1[0], popt1[1]))  
#plt.plot(np.array(X3), f(np.array(X3), popt3[0], popt3[1]))
#plt.plot(np.array(X), f(np.array(X), popt[0], popt[1])) 
#plt.plot(np.array(X4[:100]), f(np.array(X4[:100]), popt4[0], popt4[1])) 

#plt.scatter(list(RG2.keys()),list(RG2.values()),s=10,label='U2 {8_4} -  FPL')
#plt.scatter(list(RG1.keys()),list(RG1.values()),s=10,label='U1 {8_2} - FPL')
#plt.scatter(list(RG3.keys()),list(RG3.values()),s=10,label='U2 {8_4} - DDL')
#plt.scatter(list(RG.keys()),list(RG.values()),s=10,label='U1 {8_2} - DDL')
#plt.scatter(list(RG4.keys())[:100],list(RG4.values())[:100],s=10,label='SQ64')

#plt.xscale('log')
#plt.yscale('log')

#plt.xlim([0,10**6])
#plt.ylim([1,10**6])

plt.ylabel('Average Loop Size (s) [log scale]')
plt.xlabel('Radius (R) [log scale]')
plt.legend()

plt.show()

Sdfdfd
'''
##### n(R) vs R #######
'''
RG3 = pickle.load(open('Josh Code/n(R)vsR_DDLsU2_2000loops_relabel.txt','rb'))
RG = pickle.load(open('Josh Code/n(R)vsR_DDLsU1_2000loops_relabel.txt','rb'))
L = pickle.load(open('n(R)vsR_FPLsU1nU2_1M_2000loops.txt','rb'))
RG4 = pickle.load(open('n(R)vsR_SQ64_2000loops.txt','rb'))

RG1 = L[0]
RG2 = L[1]


X,Y = RR(RG)
X1,Y1 = RR(RG1)
X2,Y2 = RR(RG2)
X3,Y3 = RR(RG3)
X4,Y4 = RR(RG4)

popt, pcov = curve_fit(f, X[:30], Y[:30])   
popt1, pcov1 = curve_fit(f, X1[10:70], Y1[10:70])   
popt2, pcov2 = curve_fit(f, X2[10:50], Y2[10:50])
popt3, pcov3 = curve_fit(f, X3[:30], Y3[:30])
popt4, pcov4 = curve_fit(f, X4[5:200], Y4[5:200])
   
print(popt, popt1, popt2, popt3, popt4)

#plt.scatter(X2, Y2, s=10, label = 'U2 {8_4} - FPL')
#plt.scatter(X1, Y1, s=10, label = 'U1 {8_2} - FPL')
#plt.scatter(X3, Y3, s=10, label = 'U2 {8_4} - DDL')
#plt.scatter(X, Y, s=10, label = 'U1 {8_2} - DDL')
plt.scatter(X4, Y4, s=10, label = 'SQ64')

#plt.plot(np.array(X2[10:50]), f(np.array(X2[10:50]), popt2[0], popt2[1]))
#plt.plot(np.array(X1[10:70]), f(np.array(X1[10:70]), popt1[0], popt1[1]))  
#plt.plot(np.array(X3[:30]), f(np.array(X3[:30]), popt3[0], popt3[1]))
#plt.plot(np.array(X[:30]), f(np.array(X[:30]), popt[0], popt[1])) 
plt.plot(np.array(X4[5:200]), f(np.array(X4[5:200]), popt4[0], popt4[1])) 

plt.plot(np.array(X3[:30]), f(np.array(X3[:30]),-3.3, popt3[1]),'-r') 

DDLs = 3.3
FPLs = 3.65
SQ = 3.2

plt.scatter(list(RG2.keys()),list(RG2.values()),s=10,label='U2 {8_4} -  FPL')
plt.scatter(list(RG1.keys()),list(RG1.values()),s=10,label='U1 {8_2} - FPL')
plt.scatter(list(RG3.keys()),list(RG3.values()),s=10,label='U2 {8_4} - DDL')
plt.scatter(list(RG.keys()),list(RG.values()),s=10,label='U1 {8_2} - DDL')
plt.scatter(list(RG4.keys()),list(RG4.values()),s=10,label='SQ64')

plt.xscale('log')
plt.yscale('log')

plt.xlim([10**(-2),10**4])
plt.ylim([10**(-12),0])

#plt.xlim([-2,5])
#plt.ylim([-16,0])

plt.xlabel('Radius (R)')
plt.ylabel('Average number density of loops n(R)')
plt.legend()

plt.show()
'''

##### P(s) vs s #######

PLs3 = pickle.load(open('../DDLs/P(s)vss_DDLsU2_20000loops_relabel.txt','rb'))
PLs = pickle.load(open('../DDLs/P(s)vss_DDLsU1_4000loops_relabel.txt','rb'))
L = pickle.load(open('../I=4/ET/P(s)vss_FPLsU1nU2_4000n20000loops.txt','rb'))
PLs4 = pickle.load(open('../../Square/P(s)vss_SQ64_2000loops.txt','rb'))

PLs1 = L[0]
PLs2 = L[1]

X,Y = RR(PLs)
X1,Y1 = RR(PLs1)
X2,Y2 = RR(PLs2)
X3,Y3 = RR(PLs3)
X4,Y4 = RR(PLs4)

s1,inter1,_,_,se1 = linregress(X1[5:60], Y1[5:60])
print(s1,inter1,se1)
s2,inter2,_,_,se2 = linregress(X2[5:30], Y2[5:30])
print(s2,inter2,se2)
s,inter,_,_,se = linregress(X[3:25], Y[3:25])
print(s,inter,se)
s3,inter3,_,_,se3 = linregress(X3[3:25], Y3[3:25])
print(s3,inter3,se3)
s4,inter4,_,_,se4 = linregress(X4[:200], Y4[:200])
print(s4,inter4,se4)


popt, pcov = curve_fit(f, X[1:20], Y[1:20])   
popt1, pcov1 = curve_fit(f, X1[5:40], Y1[5:40])   
popt2, pcov2 = curve_fit(f, X2[5:30], Y2[5:30])
popt3, pcov3 = curve_fit(f, X3[2:25], Y3[2:25])
   
print(popt1, popt2) #, popt, popt3,popt4)

plt.scatter(X2, Y2, s=10, label = 'U2 {8_4} - FPL')
plt.scatter(X1, Y1, s=10, label = 'U1 {8_2} - FPL')
#plt.scatter(X3, Y3, s=10, label = 'U2 {8_4} - DDL')
#plt.scatter(X, Y, s=10, label = 'U1 {8_2} - DDL')
#plt.scatter(X4, Y4, s=10, label = 'SQ64')
 
plt.plot(np.array(X2[5:30]), f(np.array(X2[5:30]), popt2[0], popt2[1]))
plt.plot(np.array(X1[5:40]), f(np.array(X1[5:40]), popt1[0], popt1[1]))  
#plt.plot(np.array(X3[2:25]), f(np.array(X3[2:25]), popt3[0], popt3[1])) 
#plt.plot(np.array(X[1:20]), f(np.array(X[1:20]), popt[0], popt[1])) 
#plt.plot(np.array(X4[2:130]), f(np.array(X4[2:130]), popt4[0], popt4[1]),'black') 
'''
fig = plt.figure()

#plt.axvline(x = 13.5, color = 'grey', label = 'Grey lines are log-periodic')
#plt.axvline(x = 244, color = 'grey')
#plt.axvline(x = 4787, color = 'grey')

#plt.axvline(x = 13.5, color = 'grey', label = 'Grey lines are log-periodic')
#plt.axvline(x = 115, color = 'grey')
#plt.axvline(x = 1082, color = 'grey')

#plt.scatter(list(PLs4.keys()),list(PLs4.values()),s=5,label='Square, size=64X64')
#plt.plot(np.exp(np.array(X4[:200])), np.exp(f(np.array(X4[:200]), s4, inter4)),linewidth='1') 

plt.plot(np.exp(np.array(X2[5:30])), np.exp(f(np.array(X2[5:30]), s2, inter2)),linewidth='1') 
plt.scatter(list(PLs2.keys()),list(PLs2.values()),s=3,label='U2 {8_4} -  FPL')

plt.plot(np.exp(np.array(X1[5:60])), np.exp(f(np.array(X1[5:60]), s1, inter1)),linewidth='1') 
plt.scatter(list(PLs1.keys()),list(PLs1.values()),s=3,label='U1 {8_2} - FPL')

#plt.plot(np.exp(np.array(X3[3:25])), np.exp(f(np.array(X3[3:25]), s3, inter3)),linewidth='1') 
#plt.scatter(list(PLs3.keys()),list(PLs3.values()),s=3,label='U2 {8_4} - DDL')

#plt.plot(np.exp(np.array(X[3:25])), np.exp(f(np.array(X[3:25]), s, inter)),linewidth='1') 
#plt.scatter(list(PLs.keys()),list(PLs.values()),s=3,label='U1 {8_2} - DDL')

plt.xscale('log')
plt.yscale('log')
'''
#plt.xlim([1,10**7])
#plt.ylim([10**(-15),10**9])

#plt.xlim([1,10**5])
#plt.ylim([10**(-10),10**3])

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

plt.xlabel('Loop Lengths (s)')
plt.ylabel('Density of loops of length s, P(s)')

plt.legend()

plt.show()




