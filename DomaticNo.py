import numpy as np
import networkx as nx
from itertools import groupby
import matplotlib.pylab as plt
from matplotlib.pyplot import pause
from scipy.spatial import KDTree
import pickle
import math
import pylab
import collections as cl

G = pickle.load(open('../I=2/VC/ContractedTilling_VC_I=2.gpickle','rb'))
pos = pickle.load(open('../I=2/VC/CO_NodesPositions_VC_I=2.txt','rb'))

N = list(G.nodes())
for i in N:
    if(np.linalg.norm(pos[i]-(0,0))>2.7):
        G.remove_node(i)

#NC = {3:'green',142:'blue',101:'red',102:'orange'}
NC = {3:'red',142:'blue',101:'green',102:'orange'}
Nn = {102:(61,62),62:(21,22),22:(10,302),302:(261,262),262:(221,222),222:(181,182)}

Num = [NC]
for i,j in Nn.items():
    NumF = []
    for k in Num:
        C = list(NC.values())
        C.remove(NC[3])
        C.remove(k[i])
        d1 = k.copy()
        d1[j[0]] = C[0]
        d1[j[1]] = C[1]
        d2 = k.copy()
        d2[j[1]] = C[0]
        d2[j[0]] = C[1]
        NumF.append(d1)
        NumF.append(d2)
    Num = NumF

Numf = []
for i in Num:
    if(i[182]!='red' and i[182]!='blue'):       #'red'
        Numf.append(i)

NumF = []
n3 = list(G.neighbors(3))
for i in Numf:
    i[141] = list(set(NC.values())-set([i[142],i[182],i[3]]))[0]
    c = []
    for j in n3:
        c.append(i[j])
    if(len(set(c))==3):
        for k in range(len(c)):
            if(len(set([c[k],c[k-len(c)+1],c[k-len(c)+2],c[k-len(c)+3]])) == 1):
                flag = False
                break
            else:
                flag = True
        if flag:
            NumF.append(i)
#print(len(NumF))

nf = []
nx4 = []
for i in NumF:
    visited = []
    for k in range(len(n3)):
        if (len(set([i[n3[k]], i[n3[k - len(n3) + 1]], i[n3[k - len(n3) + 2]]])) == 1):
            flag = True
            x = [n3[k],n3[k - len(n3) + 1],n3[k - len(n3) + 2]]
            break
        else:
            flag = False
    if flag:
        n = list(G.neighbors(x[1]))
        n.remove(3)
        visited.extend(n)
        sA = set(G.neighbors(n[0])).difference(set(n3))
        sB = set(G.neighbors(n[1])).difference(set(n3))
        a = list(sA.intersection(sB))[0]
        b = list(sA.difference(sB))[0]
        c = list(sB.difference(sA))[0]
        i[a] = 'red'    #'red'
        i[b] = list({'red','blue','green','orange'}.difference({'red',i[n[0]],i[x[1]]}))[0]        #'red'
        i[c] = list({'red','blue','green','orange'}.difference({'red', i[n[1]], i[x[1]]}))[0]      #'red'
        visited.extend([a,b,c])
        col = [i[c]]
        while len(visited)<16:
            n = G.neighbors(c)
            for k in n:
                if(G.degree(k)==4 and k not in visited):
                    col.append(i[k])
                    break
            n = set(G.neighbors(k))
            n1 = list(n.intersection(set(n3)))
            col.extend([i[n1[0]],i[n1[1]]])
            l = list(n.difference(set(visited)|set(n3)))
            if(l==[]):
                break
            elif(len(set(col)) == 3):
                c = l[0]
                i[c] = list({'red','blue','green','orange'}.difference(set(col)))[0]
                col = [i[c]]
                visited.append(c)
                visited.append(k)
            elif(len(set(col)) == 4):
                c = l[0]
                col = []
                visited.append(c)
                visited.append(k)
            elif(len(set(col)) == 2):
                visited.remove(c)
                c = b
                col = [i[c]]
    else:
        nx4.append(i.copy())
        l1 = [i.copy()]
        no = []
        for k in range(len(n3)):
            if (len(set([i[n3[k]], i[n3[k - len(n3) + 1]]])) == 1):
                ll = list(set(G.neighbors(n3[k])).intersection(G.neighbors(n3[k - len(n3) + 1])))
                ll.remove(3)
                a,b = set(G.neighbors(ll[0])).difference(set(n3))
                no.extend([a,b])
                visited.extend([a,b,ll[0]])
                c1,c2 = list({'red', 'blue', 'green', 'orange'}.difference(set([ i[ll[0]], i[n3[k]], i[n3[k - len(n3) + 1]] ])))
                l2 = []
                for j in l1:
                    k1 = j.copy()
                    k1[a] = c1
                    k1[b] = c2
                    k2 = j.copy()
                    k2[a] = c2
                    k2[b] = c1
                    l2.append(k1)
                    l2.append(k2)
                l1 = l2
        n4 = [62,102,142,182,222,262,302,22]
        visit = visited.copy()
        for k1 in l2:
            visited = visit.copy()
            for j in no:
                col = [k1[j]]
                n = G.neighbors(j)
                for k in n:
                    if (G.degree(k) == 4 and k not in visited):
                        col.append(k1[k])
                        break
                n = set(G.neighbors(k))
                n1 = list(n.intersection(set(n3)))
                col.extend([k1[n1[0]], k1[n1[1]]])
                l = list(n.difference(set(visited) | set(n3)))
                if (len(set(col)) == 3):
                    c = l[0]
                    k1[c] = list({'red', 'blue', 'green', 'orange'}.difference(set(col)))[0]
                    visited.append(c)
                    visited.append(k)
                elif (len(set(col)) == 4):
                    visited.append(l[0])
                    visited.append(k)
            n4_re = set(n4).difference(set(visited))
            p = 0
            for j in n4_re:
                ll = list(set(G.neighbors(j)).difference(set(k1.keys())))
                if(len(ll)==2):
                    k11 = k1.copy()
                    k11[ll[0]] = 'red'   #'red'
                    k12 = k1.copy()
                    k12[ll[1]] = 'red'   #'red'
                    nf.extend([k11,k12])
                    p = 1
            if(p==0):
                nf.extend([k1])
#print(len(nf))
numF = []
for i in NumF:
    if(i not in nx4):
        numF.append(i)
numF.extend(nf)
GnumF = pickle.load(open('../../../../DomNum/green_W0.txt','rb'))
#numF.extend(GnumF)
print(len(numF))

# for i in numF:
#     node_colors = [i[n] if n in i.keys() else 'gray' for n in G.nodes]
#     nx.draw(G,pos=pos,node_size=160,node_color=node_colors) #,with_labels=True)
#     #plt.savefig('../../../../DomNum/Ex_%i.pdf'%c)
#     plt.show()

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def show(i,j,r,u,v,u_,v_):
    theta = np.arccos(round(np.dot(pos[v]-pos[3],pos[5]-pos[3])/(np.linalg.norm(pos[v]-pos[3])*np.linalg.norm(pos[5]-pos[3])),2))
    if ((pos[v]-pos[3])[0] * (pos[5]-pos[3])[1] - (pos[v]-pos[3])[1] * (pos[5]-pos[3])[0] < 0):
        theta = -theta
    pos1 = {}
    for n,p in pos.items():
        pos1[n] = rotate(pos[3], p, theta)

    theta = np.arccos(round(np.dot(pos[u_]-pos[3], pos[164]-pos[3]) / (np.linalg.norm(pos[u_]-pos[3]) * np.linalg.norm(pos[164]-pos[3])),2))
    if ((pos[u_]-pos[3])[0] * (pos[164]-pos[3])[1] - (pos[u_]-pos[3])[1] * (pos[164]-pos[3])[0] < 0):
        theta = -theta
    pos2 = {}
    for n, p in pos.items():
        pos2[n] = rotate(pos[3], p, theta) + np.array((4.4,0))

    pos3 = {}
    for n, p in pos.items():
        pos3[n] = pos[n] + np.array((2.2,-5.2))

    # pylab.ion()
    # plt.clf()
    node_colors = [i[n] if n in i.keys() else 'gray' for n in G.nodes]
    W = [160 if n in i.keys() else 10 for n in G.nodes]
    nx.draw(G, pos=pos1, node_size=W, node_color=node_colors) #, with_labels=True)
    node_colors = [j[n] if n in j.keys() else 'gray' for n in G.nodes]
    W = [160 if n in j.keys() else 10 for n in G.nodes]
    nx.draw(G, pos=pos2, node_size=W, node_color=node_colors) #, with_labels=True)
    node_colors = [r[n] if n in r.keys() else 'gray' for n in G.nodes]
    W = [160 if n in r.keys() else 10 for n in G.nodes]
    nx.draw(G, pos=pos3, node_size=W, node_color=node_colors) #, with_labels=True)
    # pause(0.05)
    # pylab.show()
    plt.show()

BV4 = [44,4,5,244,204,164,124,84]
Pairs = []
for k in range(len(BV4)):
    Pairs.append((BV4[k], BV4[k-len(BV4) +1]))

cc = 0
for i in numF:
    cP = {(u,v):(i[u],i[v]) for u,v in Pairs if u in i.keys() and v in i.keys()}   #can also include not in i,j
    for j in GnumF[:]:
        cP_ = {(v,u): (j[v], j[u]) for u, v in Pairs if u in j.keys() and v in j.keys()}
        com = list( set(cP.values()).intersection(set(cP_.values())) )
        for k in com:
            visited = []
            u,v = list(cP.keys())[list(cP.values()).index(k)]
            v_,u_ = list(cP_.keys())[list(cP_.values()).index(k)]
            n = [q for q in G.neighbors(v) if G.degree(q)==4]
            n1 = [q for q in G.neighbors(u_) if G.degree(q) == 4]
            t3 = []  #{}
            n6 = [i[v]]
            n6_1 = []
            for q in n:
                n6.append(i[q])
                ll = list(set(G.neighbors(q)).difference(set(n3)|set([q,v])))    #set(visited)))
                if (ll[0]!=u and ll[0] in i.keys()):
                    t3.append(i[ll[0]])   #[ll[0]] = i[ll[0]]
                elif (ll[0]!=u and ll[0] not in i.keys()):
                    t3.append('dof')      #[ll[0]] = 'dof'
                elif (ll[0]==u):
                    n6_1 = [i[pp] for pp in G.neighbors(u) if pp in i.keys()]
                    n6_1.append(i[u])

            for q in n1:
                n6.append(j[q])
                ll = list(set(G.neighbors(q)).difference(set(n3)|set([q,u_])))   #set(visited)))
                if (ll[0]!=v_ and ll[0] in j.keys()):
                    t3.append(j[ll[0]])       #[ll[0]] = j[ll[0]]
                elif (ll[0]!=v_ and ll[0] not in j.keys()):
                    t3.append('dof')          #[ll[0]] = 'dof'
                elif (ll[0]==v_):
                    n6_1 = [j[pp] for pp in G.neighbors(v_) if pp in j.keys()]
                    n6_1.append(j[v_])
            n6_re = {'red','green','blue','orange'}.difference(set(n6))
            #t3v = list(t3.values())
            if((t3[0] == t3[1] and  t3[0] in n6_re) or n6[0] in t3):
                continue
            else:
                #print(t3,n6_re)
                # t3k = list(t3.keys())
                # p1 = pos[84]
                # p2 = pos[44]
                # t3_1n = [i[kk] for kk in G.neighbors(t3k[0])]
                # t3_2n = [j[kk] for kk in G.neighbors(t3k[1])]
                show(i, j, i, u, v, u_, v_)
                # if('red' in set(n6_1)):
                #     #show(i, j, i, u, v, u_, v_)
                #     cc += 1
                #     for r in numF:
                #         if(84 in r.keys() and 44 in r.keys() and r[84]=='red' and r[44]=='red'):
                #             #cc += 0
                #             show(i, j, r, u, v, u_, v_)
                #         elif(84 in r.keys() and 44 not in r.keys() and r[84]=='red'):
                #             #cc += 0
                #             show(i,j,r,u,v,u_,v_)
                #         elif (44 in r.keys() and 84 not in r.keys() and r[44] == 'red'):
                #             #cc += 0
                #             show(i, j, r, u, v, u_, v_)
print(cc)
# c=1
# for i in numF[:1]:
#     node_colors = [i[n] if n in i.keys() else 'gray' for n in G.nodes]
#     nx.draw(G,pos=pos,node_size=160,node_color=node_colors,with_labels=True)
#     #plt.savefig('../../../../DomNum/Ex_%i.pdf'%c)
#     c += 1
#     plt.show()




