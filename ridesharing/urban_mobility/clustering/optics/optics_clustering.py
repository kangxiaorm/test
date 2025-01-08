# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:55:02 2019

@author: Administrator
"""
import numpy as np 
from gisdata.distance import gps_distance
from sklearn.cluster import OPTICS

def extract_hierarchy_tree(h_label):
    cls0=[np.arange(c[0],c[1]+1) for c in h_label]
    cls1=[set(c) for c in cls0]
    
    root=ClusterTree(start=h_label[-1][0],end=h_label[-1][1],sets=cls1[-1],cluster=list(cls1[-1])) 
    root.id=str(root.depth)
    cls2=cls1.copy()
    t=cls2.pop()
    
    size=len(cls2)
    n=0
    while len(cls2)>0:
        n+=1
        print("complete {0} % surplus process: {1}".format(n/size*100,size-n))
        parent=root
        c=cls2.pop()
        if c.issubset(parent.sets):
            
            subsets=[ch for ch in parent.children if c.issubset(ch.sets)]
        
            pa=parent
            while len(subsets)>0:
                pa=subsets[0]
                subsets=[ch for ch in pa.children if c.issubset(ch.sets)]
            
            
               
            indices=list(c)
            node=ClusterTree(pa,start=indices[0],end=indices[-1],sets=c,id_="{0}.{1}".format(pa.id,len(pa.children)+1))
            pa.children.append(node)
            
        else:
            pass
       
            
    
    return parent
    
                

class ClusterTree(object):
    
    
    def __init__(self,parent=None,start=-1,end=-1,children=[],object_=None,id_='',**kwargs):
        self.parent=parent
        self.start=start
        self.end=end
        self.children=children
        self.object=object_
        self.id=id_
        self.__dict__.update(kwargs)
    
    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False
    @property
    def depth(self):
        if self.parent is None:
            return 1
        else:
            return self.depth+1
        
    @property
    def root_node(self):
        if self.parent is not None:
            return self.parent.root_node
        else:
            return self
    
    
    def get_children(self):
        return self.children
    
    def insert(self,node):
        self.children.append(node)
        
    def insert_into_tree(self,node):
        pass
    
    def is_leaf(self):
        if len(self.children) == 0 :
            return True
        else:
            return False
    
    def get_leaves(self):
        pass
    
    
class OpticsHierachicalClustering(object):
    
    
    def __init__(self):
        pass
    
    def clustering(self,data,min_samples=5,max_eps=np.inf,metric=gps_distance,p=2,cluster_method='xi',xi=0.05,predecessor_correction=True,
                   min_cluster_size=1,algorithm='auto',leaf_size=30):
        c=OPTICS(min_samples,max_eps,metric,p=p,cluster_method=cluster_method,predecessor_correction=predecessor_correction,
                 min_cluster_size=min_cluster_size,algorithm=algorithm,leaf_size=leaf_size)
        print("initialized optics model")
        c.fit(data)
        print("clustering completed!")
        self.cluster_model=c
        return c
    
    def hierarchy_cluster_tree(self):
        cls=self.cluster_model.cluster_hierarchy_
        
        return extract_hierarchy_tree(cls)
        