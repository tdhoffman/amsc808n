import numpy as np
from queue import LifoQueue

WHITE = 1
GRAY  = 2
BLACK = 3

class Node:
    def __init__(self, label):
        self.label = label
        self.clear()
    
    def clear(self):
        self.d = np.inf
        self.color = WHITE
        self.parent = None


class Graph:
    def __init__(self, n, p):
        self.n, self.p = n, p
        self.dfs_ran = False
        self.vertices = [Node(i) for i in range(n)]

        probs = np.random.random(size=(n, n))
        probs = np.where(probs < p, np.ones(shape=probs.shape), np.zeros(shape=probs.shape))
        self.adj = np.triu(probs) + np.tril(probs.T, k=1)

    def get_node(self, i):
        for node in self.vertices:
            if node.label == i: return node
        raise RuntimeError('no node found')            

    def DFS(self):
        self.dfs_ran = True

        def DFSVisit(u, time, sz):
            sz += 1
            time += 1
            u.d = time
            u.color = GRAY
            
            idxs = np.argwhere(self.adj[u.label, :] > 0)
            for j in idxs:
                if j == u.label: continue
                v = self.get_node(j)
                
                if v.color == WHITE:
                    v.parent = u
                    sz, time = DFSVisit(v, time, sz)
            
            u.color = BLACK
            time += 1
            return sz, time

        for u in self.vertices: u.clear()

        time = 0

        roots = {}
        for u in self.vertices:
            if u.color == WHITE:
                sz = 0
                sz, time = DFSVisit(u, time, sz)
                roots[u] = sz
        
        self.roots = roots
        return roots
    
    def BFS(self, s):

        for u in self.vertices: u.clear()
        s = Node(s)

        s.color = GRAY
        s.d = 0
        Q = LifoQueue()
        Q.put(s)

        while not Q.empty():
            u = Q.get()
            for j in range(self.adj.shape[1]):
                if j == u.label: continue
                v = self.get_node(j)
                
                if v.color == WHITE and self.adj[u.label, j] > 00:
                    v.color = GRAY
                    v.d = u.d + 1
                    v.parent = u
                    Q.put(v)
                
            u.color = BLACK
        
         
