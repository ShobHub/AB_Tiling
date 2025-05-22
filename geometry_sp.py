'''
Jerome Lloyd: jeromeflloyd@gmail.com
date created: 2020.06.19
------------------------
base geometry for Ammann-Beenker tilings: defines prototiles, inflation rules, tilings.
------------------------
'''

import sys
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import networkx as nx
import itertools 
import copy
import os, psutil

np.set_printoptions(threshold=sys.maxsize)
pi = np.pi
s_mean = 1+np.sqrt(2)
eps = 1/10000  # overlap tolerance


class Prototile():

    '''
    bipartite geometry for the two Ammann-Beenker prototiles, square and pi/4-rhomb.
    each prototile graph is represented by adjacency matrix, which can be arranged into bipartite block matrix.
    inflations are done at the matrix level, and networkx makes graph direct from final adjacency matrix (in the graph module).
    '''

    def __init__(self, shape, COM=[0,0], theta=0, zero_charge=0):

        if shape == 't': shape = 'tri'  # shortcut aliases
        if shape == 'r': shape = 'rhomb'
        self.shape = shape
        self.COM, self.theta = COM, theta
        self.ncharge = np.array([2,2])  # (number of type-1 (black) charge, number of type-2 (white) charge)
        self.nnodes = sum(self.ncharge)  # total number of nodes
        self.eightsgen = dict()

        getattr(self, self.shape)(zero_charge)

    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def square(self, zero_charge=0):

        self.adj = np.array([[0,0,0,1],[1,0,1,0],[0,0,0,1],[0,1,0,0]], dtype=bool)  # adjacency matrix for square

        d = np.sqrt(2)/2
        self.pos = np.array([[0,d], [+d,0], [0,-d], [-d,0]], dtype=np.double)  # node positions
        if self.theta != 0:
            self.rotate(self.theta)
        if np.any(self.COM) != 0:
            self.translate(self.COM)

        self.charge = [0, 1, 0, 1]  # node charges
        if zero_charge == 1:
            self.charge = [1, 0, 1, 0] 

        self.gen = [0]*4  # node inflation.gens ("generation")

        self.triangles = [(0,1,3), (2,1,3)]  # list of current triangles (half-squares)
        self.rhombs = list()  # list of current rhombs
        self.squares = [(0,1,2,3)]

        return

    def rhomb(self, zero_charge=0):

        self.adj = np.array([[0,0,0,0],[1,0,1,0],[0,0,0,0],[1,0,1,0]], dtype=bool)  # adjacency matrix for rhomb

        d = np.cos(pi/8)
        s = np.sin(pi/8)
        self.pos = np.array([[0,s], [+d,0], [0,-s], [-d,0]], dtype=np.double)  # node positions
        if self.theta != 0:
            self.rotate(self.theta)
        if np.any(self.COM) != 0:
            self.translate(self.COM)

        self.charge = [0, 1, 0, 1]  # node charges
        if zero_charge == 1:
            self.charge = [1, 0, 1, 0]

        self.gen = [0]*4  # node inflation.gens ("generations")

        self.triangles = list()  # list of current triangles (half-squares)
        self.rhombs = [(0,1,2,3)]  # list of current rhombs
        self.squares = list() 

        return

    def rotate(self, theta):
        '''rotate tile about COM, clockwise)
        '''
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.pos = self.pos@R

        return

    def translate(self, COM):
        '''translate time centre-of-mass (COM)
        '''
        self.pos += COM

        return


class Tiling(Prototile):

    '''
    tiling formed from inflation of initial seed (can be a single prototile or more complicated set).
    inflation acts by creating new adjacency matrix containing entries for each inflated triangle, rhomb 
    of previous tiling.
    '''

    def __init__(self, shape=None, k=None, inflations=0):

        self.inflations = 0
        if shape is not None:
            if shape == 'eightemp':
                self.eightemp()
            elif shape =='crown':
                self.crown()
            elif shape == 'vertexconfig':
                self.vertexconfig(k)
            elif shape == 'star':
                self.star()
            else:
                self.seed(shape)

        if inflations > 0:
            self.inflate(inflations)

    def seed(self, shape, COM=[0,0], theta=0, zero_charge=0):
        super().__init__(shape, COM, theta, zero_charge)


    def append_tile(self, shape, COM, theta, charge=0):
        '''
        append tile to existing tiling. shape is one of square, rhomb. COM and theta specify tile positioning and rotation
        (clockwise).
        if new nodes overlap old nodes, new nodes are merged into old. unmerged nodes are added to graph.
        '''
        tile = Prototile(shape, COM, theta, charge)
        self.labels = [None]*4

        for idx1, p1 in enumerate(tile.pos): # check for overlapping nodes
            for idx2, p2 in enumerate(self.pos):
                if np.abs(np.linalg.norm(p1-p2)) < eps:
                    if tile.charge[idx1] != self.charge[idx2]:  # check for charge conflict
                          charge = 1
                    self.labels[idx1] = idx2
        if charge == 1:  # correct for charge conflict
            tile = Prototile(shape, COM, theta, zero_charge=1)

        for idx, label in enumerate(self.labels):
            if label is None:
                self.labels[idx] = self.nnodes  # new node label
                self.nnodes += 1
                self.ncharge += [1^tile.charge[idx], 0^tile.charge[idx]]

                self.adj = np.c_[self.adj, np.zeros(self.adj.shape[0], dtype=np.bool)]  # add node space to adjacency mx
                self.adj = np.r_[self.adj, [np.zeros(self.adj.shape[1], dtype=np.bool).T]]
                self.pos = np.r_[self.pos, [tile.pos[idx]]]  # new node pos
                self.charge.append(tile.charge[idx])  # new node charge
                self.gen.append(self.inflations)  # new node.gen
        adj_idx = (np.array(self.labels)[:, np.newaxis], np.array(self.labels))
        self.adj[adj_idx] = tile.adj  # copy tile connectivity to adjancency mx

        if shape == 'square':
            self.triangles += [(self.labels[0], self.labels[1], self.labels[3]), (self.labels[2], self.labels[1], self.labels[3])]
        else:
            self.rhombs.append((self.labels[0], self.labels[1], self.labels[2], self.labels[3]))

        return

    def to_sparse(self):
        adj_ = []
        for row in self.adj:
            rowvals = []
            for idx, col in enumerate(row):
                if col != 0:
                    rowvals.append(idx)
            adj_.append(rowvals)
        self.adj = adj_

    def to_dense(self):
        adj_ = np.zeros((len(self.adj), len(self.adj)), dtype=bool)
        for idx, vals in enumerate(self.adj):
            for v in vals:
                adj_[idx, v] = 1
        self.adj = adj_

    def remove_non_bipartite(self):
        for idx, node_list in enumerate(self.adj):
            charge0 = self.charge[idx]
            for node in node_list:
                if self.charge[node] == charge0:
                    self.adj[idx].remove(node)


    def decorate (self):

        def rhomb_geoDec(r,a):
            
            #adj_inflate=[[19,26,27,18],[16,15],[13,32,34,12],[21,10],[20,22],[22,11],[37,14],[17,37],[36,25,33,35],[36,33,31,28],[5],[35],[11],[14],[31],[6],[7],[28],[17],[20],[25],[4],[],[24,39],[4],[],[8],[9],[],[7],[6],[],[9],[],[8],[],[],[],[29,30],[5]]

            adj_decorate=[[],[],[],[],[23],[23],[38],[38],[24,39],[30,29],[],[39],[],[],[30],[],[],[29],[],[],[24],[],[10,21],[],[],[26,19],[],[],[18,27],[],[],[32,13],[],[34,32],[],[34,12],[27,26],[16,15],[],[]]

            adj_full=[[19,26,27,18],[16,15],[13,32,34,12],[21,10],[20,22,23],[22,23,11],[37,38,14],[17,38,37],[36,25,24,39,33,35],[30,29,36,33,31,28],[],[35,39],[],[],[30,31],[],[],[28,29],[],[],[24,25],[],[10,21],[24,39],[],[26,19],[],[],[18,27],[],[],[32,13],[],[34,32],[],[34,12],[27,26],[16,15],[29,30],[]]

            adjD_matrix=[]  #np.zeros((len(adj_decorate),len(adj_decorate)))
            for i in range(len(adj_decorate)):
                for j in adj_decorate[i]:
                    #adjD_matrix[i,j] = 1
                    adjD_matrix.append((i+40*a,j+40*a))

            adjF_matrix=[]  #np.zeros((len(adj_full),len(adj_full)))
            for i in range(len(adj_full)):
                for j in adj_full[i]:
                    #adjF_matrix[i,j] = 1
                    adjF_matrix.append((i+40*a,j+40*a))

            posR={0: np.array([0., 0.]), 27: np.array([ 0.38, -0.92]), 26: np.array([-0.38, -0.92]), 19: np.array([-0.92, -0.38]), 18: np.array([ 0.92, -0.38]), 1: np.array([ 5.38, -2.23]), 15: np.array([ 4.46, -2.61]), 16: np.array([ 4.46, -1.85]), 2: np.array([ 0.  , -4.46]), 34: np.array([-0.38, -3.54]), 32: np.array([ 0.38, -3.54]), 12: np.array([-0.92, -4.08]), 13: np.array([ 0.92, -4.08]), 3: np.array([-5.38, -2.23]), 21: np.array([-4.46, -1.85]), 10: np.array([-4.46, -2.61]), 4: np.array([-3.15, -1.31]), 22: np.array([-3.54, -2.23]), 23: np.array([-2.77, -2.23]), 20: np.array([-2.23, -0.92]), 5: np.array([-3.15, -3.15]), 11: np.array([-2.23, -3.54]), 6: np.array([ 3.15, -3.15]), 38: np.array([ 2.77, -2.23]), 37: np.array([ 3.54, -2.23]), 14: np.array([ 2.23, -3.54]), 7: np.array([ 3.15, -1.31]), 17: np.array([ 2.23, -0.92]), 8: np.array([-0.92, -2.23]), 39: np.array([-1.85, -2.61]), 24: np.array([-1.85, -1.85]), 36: np.array([ 0.  , -1.85]), 33: np.array([ 0.  , -2.61]), 25: np.array([-1.31, -1.31]), 35: np.array([-1.31, -3.15]), 9: np.array([ 0.92, -2.23]), 29: np.array([ 1.85, -1.85]), 30: np.array([ 1.85, -2.61]), 28: np.array([ 1.31, -1.31]), 31: np.array([ 1.31, -3.15])}

            pos1_0,pos1_1,pos1_2,pos1_3 = self.pos[np.array(r)]*((1+np.sqrt(2))**2)
            axis = (pos1_1-pos1_3)/np.linalg.norm((pos1_1-pos1_3)) 
            #theta=-np.arccos(np. dot(axis,(1,0)))
            if(np.cross(pos1_0-pos1_3, pos1_2-pos1_3) < 0):
                theta=-np.arccos(np. dot(axis,(1,0)))
            else:
                theta=-np.arccos(np. dot(axis,(1,0)))+np.pi
            if(np.abs(axis[1])>1e-10 and axis[1]<0):
                theta=-theta+(2*np.pi)            


            '''plt.plot(pos1_0[0],pos1_0[1],marker="o", markersize=6,markerfacecolor="black")
            plt.plot(pos1_1[0],pos1_1[1],marker="o", markersize=6,markerfacecolor="green")
            plt.plot(pos1_2[0],pos1_2[1],marker="o", markersize=6,markerfacecolor="pink")
            plt.plot(pos1_3[0],pos1_3[1],marker="o", markersize=6,markerfacecolor="yellow")'''

            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            posR = dict((id, val@R) for id, val in posR.items())

            pos_decorate = dict((id, np.round(pos1_0,2)+val) for id, val in posR.items())
            
            return adjD_matrix, adjF_matrix, pos_decorate

        
        def triangle_geoDec(t,a):

            #adj_inflate=[[15,28,14],[11,12],[17,8],[16,29],[9,21,18],[25,10],[13,25],[20,29,27,22],[4],[10],[],[5],[6],[27],[13],[3],[18],[16],[],[3],[19],[7],[],[5],[23,26],[],[6],[],[7],[]]

            adj_decorate=[[],[],[],[20],[19],[24],[24],[26,23],[],[],[23],[],[],[26],[],[],[19],[],[17,8],[],[21],[],[21,9],[],[],[12,11],[],[28,14],[],[15,28]]

            adj_full=[[15,28,14],[11,12],[17,8],[16,20,29],[9,19,21,18],[25,24,10],[13,24,25],[20,26,23,29,27,22],[],[],[22,23],[],[],[27,26],[],[],[18,19],[],[17,8],[],[19,21],[],[21,9],[],[23,26],[12,11],[],[28,14],[],[15,28]]

            adjD_matrix=[]   #np.zeros((len(adj_decorate),len(adj_decorate)))
            for i in range(len(adj_decorate)):
                for j in adj_decorate[i]:
                    #adjD_matrix[i,j] = 1
                    adjD_matrix.append((i+30*a+40*len(list(self.rhombs)),j+30*a+40*len(list(self.rhombs))))

            adjF_matrix=[]  #np.zeros((len(adj_full),len(adj_full)))
            for i in range(len(adj_full)):
                for j in adj_full[i]:
                    #adjF_matrix[i,j] = 1
                    adjF_matrix.append((i+30*a+40*len(list(self.rhombs)),j+30*a+40*len(list(self.rhombs))))

            unit = lambda posX, posY, posZ: posX+(posZ-posY)/(5.828*np.linalg.norm(posZ-posY))
   
            pos0,pos1,pos2 = self.pos[np.array(t)]    #*((1+np.sqrt(2))**2)

            u7=unit(unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos1),pos0,pos2)
            u3=unit(unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos2),pos1,pos2)
            u4=unit(unit(unit(u3,pos0,pos2),pos0,pos1),pos0,(pos1+pos2)/2)
            u6=unit(unit(unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos1),pos2,pos1),pos0,pos1)


            posD={0:pos0, 1:pos1, 2:pos2, 3:unit(unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos2),pos1,pos2), 4:unit(unit(unit(u3,pos0,pos2),pos0,pos1),pos0,(pos1+pos2)/2), 5:unit(unit(u6,pos0,(pos1+pos2)/2),pos0,pos2), 6:unit(unit(unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos1),pos2,pos1),pos0,pos1), 7:unit(unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos1),pos0,pos2), 8:unit(pos2,pos2,pos1), 9:unit(u4,pos2,pos1), 10:unit(unit(u7,pos0,(pos1+pos2)/2),pos0,pos1), 11:unit(pos1,pos1,pos2), 12:unit(pos1,pos1,pos0), 13:unit(unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos1),pos2,pos1), 14:unit(pos0,pos0,pos1), 15:unit(pos0,pos0,pos2), 16:unit(u3,pos0,pos2), 17:unit(pos2,pos2,pos0), 18:unit(unit(u3,pos0,pos2),pos0,(pos1+pos2)/2), 19:unit(unit(u3,pos0,pos2),pos0,pos1), 20:unit(u7,pos1,pos2), 21:unit(unit(u7,pos1,pos2),pos0,(pos1+pos2)/2), 22:unit(u7,pos0,(pos1+pos2)/2), 23:unit(u7,pos0,pos1), 24:unit(unit(u7,pos0,pos1),pos2,pos1), 25:unit(u6,pos0,(pos1+pos2)/2), 26:unit(u7,pos2,pos1), 27:unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos1), 28:unit(pos0,pos0,(pos1+pos2)/2), 29:unit(unit(pos0,pos0,(pos1+pos2)/2),pos0,pos2)} #unit(unit(unit(unit(pos0,pos0,pos1),pos0,pos1),pos1,pos2),pos1,pos0)

            pos_decorate = dict((id, np.round(val*((1+np.sqrt(2))**2),2)) for id, val in posD.items())
            
            return adjD_matrix, adjF_matrix, pos_decorate
        
        #gr = nx.Graph()
        Pos_Decorate={}
        c=0
        BN_r=[0,18,17,7,16,1,15,6,14,13,2,12,11,5,10,3,21,4,20,19]
        PR=[[3,10,22,5,23,39,11,35,12,2],[2,13,31,14,30,38,6,37,15,1],[1,16,37,7,38,29,17,28,18,0],[0,19,25,20,24,23,4,22,21,3]]
        bound_nodes={}
        P=[]
        Adj_Decorate=iter([]) #[[]]
        Adj_Full=iter([])  #[[]]
        
        for r in self.rhombs:
            AD,ADF,PD=rhomb_geoDec(r,c)
            if(c==0):
                Adj_Decorate = itertools.chain(Adj_Decorate,AD)
                Adj_Full = itertools.chain(Adj_Full,ADF)
                #Adj_Decorate.extend(AD)          #=block_diag(AD)
                #Adj_Full.extend(ADF)             #=block_diag(ADF)
            else:
                Adj_Decorate = itertools.chain(Adj_Decorate,AD)
                Adj_Full = itertools.chain(Adj_Full,ADF)
                #Adj_Decorate.extend(AD)          #=block_diag(Adj_Decorate,AD)
                #Adj_Full.extend(ADF)              #=block_diag(Adj_Full,ADF)

            for i in range(40):
                Pos_Decorate[i+40*c]=PD[i] 
                if(i in BN_r):
                    bound_nodes[i+40*c]=PD[i]  
 
            for j in PR:
                P.append([k+40*c for k in j])
                    
            c=c+1
   
        c=0
        BN_t=[0,14,13,6,12,1,11,5,10,9,4,8,2,17,16,3,15]
        PT=[[2,17,18,16,19,20,3,29,15,0],[0,14,27,13,26,24,6,25,12,1]]

        for t in self.triangles:
          
            AD,ADF,PD=triangle_geoDec(t,c) 
            if(self.rhombs==[] and c==0):
                Adj_Decorate = itertools.chain(Adj_Decorate,AD)
                Adj_Full = itertools.chain(Adj_Full,ADF)
                #Adj_Decorate.extend(AD)              #=block_diag(AD)
                #Adj_Full.extend(ADF)                 #=block_diag(ADF)
            else:
                Adj_Decorate = itertools.chain(Adj_Decorate,AD)
                Adj_Full = itertools.chain(Adj_Full,ADF)
                #Adj_Decorate.extend(AD)              #=block_diag(Adj_Decorate,AD)
                #Adj_Full.extend(ADF)                  #l=block_diag(Adj_Full,ADF)

            for i in range(30):
                Pos_Decorate[i+30*c+40*len(list(self.rhombs))]=PD[i] 
                if(i in BN_t):
                    bound_nodes[i+30*c+40*len(list(self.rhombs))]=PD[i]  

            for j in PT:
                P.append([k+30*c+40*len(list(self.rhombs)) for k in j])
            c=c+1

    
        '''Adj_Decorate=np.delete(Adj_Decorate,0,axis=0)
        Adj_Decorate=np.delete(Adj_Decorate,0,axis=1)
        Adj_Full=np.delete(Adj_Full,0,axis=0)
        Adj_Full=np.delete(Adj_Full,0,axis=1)'''

        return Adj_Decorate, Adj_Full, Pos_Decorate, bound_nodes,P
        

        '''rows, cols = np.where(Adj_Decorate == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr.add_edges_from(edges)'''

        '''rows, cols = np.where(Adj_Decorate[0:24*40,0:24*40] == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr.add_edges_from(edges)
        pp={}
        c=0
        d=0
        L={}
        for i in list(Pos_Decorate.keys())[24*40:24*40+34*30]:
            pp[c]=Pos_Decorate[i]
            if(c%30==0):
                L[c]=d
                d=d+1
            else:
                L[c]=' '
            c=c+1
        #nx.draw(gr,pp,node_size=6, node_color='red',with_labels=True,labels=L)
        #plt.show()'''
        #nx.draw(gr,Pos_Decorate,node_size=2, node_color='red')
    
    def inflate(self, n=1):

        ''' inflate current tiling'''

        inflation_factor = s_mean

        if self.inflations == 0:
            Anodes = [idx for idx, charge in enumerate(self.charge) if charge == 0]
            Bnodes = [idx for idx, charge in enumerate(self.charge) if charge == 1]
            self.nedgeAA = np.count_nonzero(self.adj[(np.array(Anodes)[:, np.newaxis], np.array(Anodes))])
            self.nedgeAB = np.count_nonzero(self.adj[(np.array(Anodes)[:, np.newaxis], np.array(Bnodes))])
            self.nedgeBB = np.count_nonzero(self.adj[(np.array(Bnodes)[:, np.newaxis], np.array(Bnodes))])
            self.nedgeBA = np.count_nonzero(self.adj[(np.array(Bnodes)[:, np.newaxis], np.array(Anodes))])

        def triangle_geometry(t):
            '''
            t is triangle in form tuple(t1, t2, t3) where t are triangle nodes.
            '''
            adj_inflate = [[3,7], [5,6], [4], [2], [3,7], [4], [0], [5,6]]
#            adj_inflate = sparse.csr_matrix([[0,0,0,1,0,0,0,1], [0,0,0,0,0,1,1,0], [0,0,0,0,1,0,0,0],  # adjacency matrix for inflated triangle
#                                    [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0],
#                                    [1,0,0,0,0,0,0,0], [0,0,0,0,0,1,1,0]], dtype=bool)

            unit = lambda posX, posY, posZ: posX+(posZ-posY)/np.linalg.norm(posZ-posY)  # unit vector from X in direction Z-Y

            pos0, pos1, pos2 = self.pos[np.array(t)]
            pos_inflate = [pos0, pos1, pos2, unit(pos0, pos0, pos2), unit(pos2, pos2, pos1),  # node positions for inflated triangle
                           unit(pos1, pos1, pos2), unit(pos1, pos1, pos0),
                           unit(unit(pos1, pos1, pos0), pos1, pos2)]

            charge = self.charge[t[0]]
            charge_inflate = np.array([0,1,1,1,0,0,0,1], dtype=bool)^charge  # bipartite charge of inflated triangle

            return adj_inflate, pos_inflate, charge_inflate


        def rhomb_geometry(r):
            '''
            R is rhomb in form tuple(r1, r2, r3, r4) where r are rhomb nodes.
            '''

            adj_inflate = [[8,9],[6,7], [8,9], [4,5], [0], [2], [2], [0], [4,5], [6,7]]


#            sparse.csr_matrix([[0,0,0,0,0,0,0,0,1,1], [0,0,0,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,0,1,1],
#                                    [0,0,0,0,1,1,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0],
#                                    [0,0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0],
#                                    [0,0,0,0,0,0,1,1,0,0]], dtype=bool)

            unit = lambda posX, posY, posZ: posX+(posZ-posY)/np.linalg.norm(posZ-posY)

            pos0, pos1, pos2, pos3 = self.pos[np.array(r)]
            pos_inflate = [pos0, pos1, pos2, pos3, unit(pos3, pos3, pos0), unit(pos3, pos3, pos2),
                           unit(pos1, pos1, pos2), unit(pos1, pos1, pos0),
                           unit(unit(pos3, pos3, pos0), pos3, pos2), unit(unit(pos1, pos1, pos2), pos1, pos0)]

            charge = self.charge[r[0]]
            charge_inflate = np.array([0,1,0,1,0,0,0,0,1,1], dtype=bool)^charge

            return adj_inflate, pos_inflate, charge_inflate


        def inflate_tile(origin_nodes):

            if len(origin_nodes) == 3:
                adj_inflate, pos_inflate, charge_inflate = triangle_geometry(origin_nodes)
            else:
                adj_inflate, pos_inflate, charge_inflate = rhomb_geometry(origin_nodes)
            self.labels = list(origin_nodes).copy()
            charge = self.charge[origin_nodes[0]]

            for idx in range(len(origin_nodes), len(adj_inflate)):
                parent = None
                for i in range(len(origin_nodes)):
                    if idx in adj_inflate[i]:
                        parent = origin_nodes[i]
                        break
                #parent = origin_nodes[np.nonzero(adj_inflate[:len(origin_nodes), idx])[0][0]]  # origin node with edge to node_idx node
                parent_out_nodes = self.adj[parent]
                #parent_out_nodes = np.nonzero(self.adj[parent, :])  # connected origin out neighbours
                overlap = None
                for node in parent_out_nodes:
                    if np.linalg.norm(self.pos[node, :]-pos_inflate[idx]) < eps:
                        overlap = node
                        break
                #overlap = np.argwhere(np.linalg.nrom(self.pos[parent_out_nodes, :]-pos_inflate[idx], axis=1) < eps)
                #overlap = np.argwhere(np.linalg.norm(self.pos[parent_out_nodes, :][0]-pos_inflate[idx], axis=1) < eps)  # check for existing node

                if overlap is not None:
                    label = overlap  # if node exists, merge
                else:
                    label = self.nnodes
                    self.nnodes += 1
                    self.ncharge += np.array([1, 0])^charge_inflate[idx]
                    self.pos[label, :] = pos_inflate[idx]
                    self.charge[label] = charge_inflate[idx]
                    self.gen[label] = self.inflations

                self.labels.append(label)

            for idx, label in enumerate(self.labels):
                for i in adj_inflate[idx]:
                    if self.labels[i] not in self.adj[label]:
                        self.adj[label].append(self.labels[i])
                        self.adj[label].sort()
                        charge0 = self.charge[label]
                        charge1 = self.charge[self.labels[i]]

                        if charge0 == 0:
                            if charge1 == 0:
                                self.nedgeAA += 1
                            else:
                                self.nedgeAB += 1
                        else:
                            if charge1 == 0:
                                self.nedgeBA += 1
                            else:
                                self.nedgeBB += 1

            if len(origin_nodes) == 3:
                new_triangles = [(self.labels[7], self.labels[0], self.labels[6]), (self.labels[7], self.labels[4], self.labels[5]),
                                    (self.labels[4], self.labels[2], self.labels[3])]
                new_rhombs = [(self.labels[6], self.labels[1], self.labels[5], self.labels[7]), (self.labels[7], self.labels[4], self.labels[3], self.labels[0])]

            else:
                new_triangles = [(self.labels[8], self.labels[0], self.labels[4]), (self.labels[8], self.labels[2], self.labels[5]),
                                 (self.labels[9], self.labels[0], self.labels[7]), (self.labels[9], self.labels[2], self.labels[6])]
                new_rhombs = [(self.labels[4], self.labels[8], self.labels[5], self.labels[3]), (self.labels[7], self.labels[1], self.labels[6], self.labels[9]),
                              (self.labels[9], self.labels[2], self.labels[8], self.labels[0])]


            return new_triangles, new_rhombs


        for i in range(n):

            self.inflations += 1

            nA, nB = self.ncharge  # number black nodes, white nodes of old tiling

            Atriangles, Btriangles, Arhombs, Brhombs = [0]*4
            for t in self.triangles:
                if self.charge[t[0]] == 0:
                    Atriangles += 1
                else:
                    Btriangles += 1
            for r in self.rhombs:
                if self.charge[r[0]] == 0:
                    Arhombs += 1
                else:
                    Brhombs += 1
            #print("Atriangles: {} Btriangles: {} Arhombs: {} Brhombs: {}".format(Atriangles, Btriangles, Arhombs, Brhombs))
            nA_ = nA+self.nedgeBA+2*self.nedgeBB+Btriangles+2*Brhombs
            nB_ = nB+self.nedgeAB+2*self.nedgeAA+Atriangles+2*Arhombs
            self.nedgeAA, self.nedgeAB, self.nedgeBA, self.nedgeBB = 0, 0, 0, 0
            ncharge_ = (nA_, nB_)  # number of black nodes, white nodes of new tiling
            nnodes_ = sum(ncharge_)

            self.adj = [[] for i in range(nnodes_)] # new adjacency matrix, empty list-of-lists sparse format
            self.pos = (np.r_[self.pos, np.zeros((nnodes_-self.nnodes, 2), dtype=np.double)])*inflation_factor
            self.charge += [None]*(nnodes_-self.nnodes)
            self.gen += [None]*(nnodes_-self.nnodes)

            triangles, rhombs = list(), list()

            for r in self.rhombs:
                new_triangles, new_rhombs = inflate_tile(r)
                triangles += new_triangles
                rhombs += new_rhombs

            for t in self.triangles:
                new_triangles, new_rhombs = inflate_tile(t)
                triangles += new_triangles
                rhombs += new_rhombs
            self.triangles, self.rhombs = triangles, rhombs

            for node, adj_list in enumerate(self.adj):
                if len(adj_list) == 8:
                    if node not in self.eightsgen.keys():
                        self.eightsgen[node] = self.inflations
        
        print(len(list(self.rhombs)),len(list(self.triangles)))
        #self.decorate()
        return

    '''
    ------------------------------------------------------------------------------------
    below are premade 'patches' that are frequently useful, composed of the basic prototiles
    ------------------------------------------------------------------------------------
    '''

    def eightemp(self):

        '''generates the basic 8-empire'''

        d = np.cos(pi/8)
        s = np.sin(pi/8)
        p = s_mean/np.sqrt(2)
        self.seed('rhomb', [d,0], 0)
        for n in range(1, 8):
            self.append_tile('rhomb', [d*np.cos(n*pi/4), d*np.sin(n*pi/4)], -n*np.pi/4)
        for n in range(1, 9):
            self.append_tile('square', [p*np.cos((2*n-1)*pi/8), p*np.sin((2*n-1)*pi/8)], -(2*n-1)*np.pi/8)
            self.append_tile('rhomb', [(2*d+s)*np.cos((n-1)*pi/4), (2*d+s)*np.sin((n-1)*pi/4)], (-n+3)*np.pi/4)

        self.eightsgen[3] = 0

        return

    def star(self):

        ''' generates star configuration'''

        d = np.cos(pi/8)
        s = np.sin(pi/8)
        p = s_mean/np.sqrt(2)
        self.seed('rhomb', [d,0], 0)
        for n in range(1, 8):
            self.append_tile('rhomb', [d*np.cos(n*pi/4), d*np.sin(n*pi/4)], -n*np.pi/4)

        for n in range(1, 9):
            self.append_tile('square', [p*np.cos((2*n-1)*pi/8), p*np.sin((2*n-1)*pi/8)], -(2*n-1)*np.pi/8)
            self.append_tile('rhomb', [(2*d+s)*np.cos((n-1)*pi/4), (2*d+s)*np.sin((n-1)*pi/4)], (-n+3)*np.pi/4)

        for n in range(1,9):
            R = (2*d+2*s+np.exp(-1j*3*np.pi/8)/np.sqrt(2))*np.exp(1j*(n-1)*np.pi/4)
            self.append_tile('square', [R.real, R.imag], -(2*n-9)*np.pi/8)
            R = (2*d+2*s+np.exp(+1j*3*np.pi/8)/np.sqrt(2))*np.exp(1j*(n-1)*np.pi/4)
            self.append_tile('square', [R.real, R.imag], -(2*n-11)*np.pi/8)

        for n in range(1,9):
            self.append_tile('rhomb', [(3*d+2*s)*np.cos(n*pi/4), (3*d+2*s)*np.sin(n*pi/4)], -n*np.pi/4)

        self.eightsgen[3] = 0


        return

    def crown(self):

        ''' generates the crown configuration'''

        d = np.cos(pi/8)
        s = np.sin(pi/8)
        p = s_mean/np.sqrt(2)
        q = 1/np.sqrt(2)
        x = d*(2-np.cos(pi/4))
        y = d*np.sin(pi/4)

        self.seed('rhomb', [d,0])
        self.append_tile('square', [q*np.cos(3*pi/8), q*np.sin(3*pi/8)], -7*pi/8)
        self.append_tile('square', [q*np.cos(3*pi/8), q*np.sin(-3*pi/8)], 7*pi/8)
        self.append_tile('rhomb', [-s,0], pi/2)
        self.append_tile('rhomb', [x, y], pi/4)
        self.append_tile('rhomb', [x, -y], -pi/4)

    def vertexconfig(self, k):

        ''' generates one of the seven vertex configurations (k is connectivity)'''

        d = np.cos(pi/8)
        s = np.sin(pi/8)
        p = s_mean/np.sqrt(2)
        q = 1/np.sqrt(2)

        if k == 3:
            self.seed('square', [q,0])
            self.append_tile('rhomb', [s*np.cos(5*pi/8), s*np.sin(5*pi/8)], -pi/8)
            self.append_tile('rhomb', [s*np.cos(5*pi/8), s*np.sin(-5*pi/8)], +pi/8)
        if k == 4:
            self.seed('rhomb', [d,0])
            self.append_tile('square', [q*np.cos(3*pi/8), q*np.sin(3*pi/8)], -7*pi/8)
            self.append_tile('square', [q*np.cos(3*pi/8), q*np.sin(-3*pi/8)], +7*pi/8)
            self.append_tile('rhomb', [-s,0], pi/2)
        if k == '5a':
            self.seed('square', [q,0], pi)
            self.append_tile('rhomb', [d*np.cos(3*pi/8), d*np.sin(3*pi/8)], -3*pi/8)
            self.append_tile('rhomb', [d*np.cos(3*pi/8), d*np.sin(-3*pi/8)], +3*pi/8)
            self.append_tile('square', [q*np.cos(6*pi/8), q*np.sin(6*pi/8)], +pi/4)
            self.append_tile('square', [q*np.cos(6*pi/8), q*np.sin(-6*pi/8)], -pi/4)
        if k == '5b':
            self.seed('square', [q,0], pi)
            self.append_tile('rhomb', [d*np.cos(3*pi/8), d*np.sin(3*pi/8)], -3*pi/8)
            self.append_tile('rhomb', [d*np.cos(3*pi/8), d*np.sin(-3*pi/8)], +3*pi/8)
            self.append_tile('square', [q*np.cos(6*pi/8), q*np.sin(6*pi/8)], -5*pi/4)
            self.append_tile('square', [q*np.cos(6*pi/8), q*np.sin(-6*pi/8)], +5*pi/4)
        if k == 6:
            self.seed('rhomb', [d,0], 0)
            self.append_tile('square', [q*np.cos(3*pi/8), q*np.sin(3*pi/8)], +5*pi/8)
            self.append_tile('square', [q*np.cos(3*pi/8), q*np.sin(-3*pi/8)], -5*pi/8)
            for n in range(0, 3):
                self.append_tile('rhomb', [d*np.cos(n*pi/4+6*pi/8), d*np.sin(n*pi/4+6*pi/8)], -n*pi/4-6*pi/8)
        if k == 7:
            self.seed('square', [q,0], np.pi)
            for n in range(2, 8):
                self.append_tile('rhomb', [d*np.cos(n*pi/4-np.pi/8), d*np.sin(n*pi/4-np.pi/8)], -n*np.pi/4+np.pi/8)
        if k == 8:
        
            self.seed('rhomb', [d,0], 0)
            for n in range(1,8):
                self.append_tile('rhomb', [d*np.cos(n*pi/4), d*np.sin(n*pi/4)], -n*np.pi/4)
            
            #self.decorate()
            self.eightsgen[3] = 0


        return


if __name__ == "__main__":

    pass
