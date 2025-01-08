# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:46:21 2019

@author: Administrator
"""
import numpy as np
import math
from scipy.spatial import ConvexHull
import haversine
from shapely.geometry import Polygon,Point
import geopandas as gpd
import osmnx as ox
from sklearn.cluster import OPTICS

def gps_distance(pt1,pt2):
    r=haversine.haversine((pt1[1],pt1[0]),(pt2[1],pt2[0]),unit="m")
    return r

class geometry(object):
    
    
    def __init__(self):
        pass
    
#    
#class Polygon(object):
#    
#    
#    def __init__(self,points,type_='polygon',**kwargs):
#        self.points=points
#        self.type_=type_
#        hull=ConvexHull(points)
#        self.convex_hull=hull
#        self.left=hull.min_bound[0]
#        self.bottom=hull.min_bound[1]
#        self.right=hull.max_bound[0]
#        self.top=hull.max_bound[1]
#        self.mbr=(self.left,self.bottom,self.right,self.top)
#        
#    
#    @property
#    def center_point(self):
#        x_sum=sum([pt[0] for pt in self.points])
#        y_sum=sum([pt[1] for pt in self.points])
#        len_=len(self.points)
#        return x_sum/len_,y_sum/len_
#    
#    @property
#    def mbr_center(self):
#        return (self.left+self.right)/2,(self.bottom+self.top)/2
    

class Region(Polygon):
    
    
    def __init__(self,id_='',boundary=[],**kwargs):
        super().__init__(boundary)
        self.id=id_
        self.__dict__.update(kwargs)
    
    def contains(self,coord:list):
        return super().contains(Point(coord))
    
class GridRegion(Region):
    
    
    def __init__(self,id_='',bounds=[],**kwargs):
        if len(bounds)==4:
            left,bottom,right,top=bounds[0],bounds[1],bounds[2],bounds[3]
            self.left,self.bottom,self.right,self.top=left,bottom,right,top
            boundary=[(self.left,self.bottom),(self.right,self.bottom),(self.right,self.top),(self.left,self.top),(self.left,self.bottom)]
            super().__init__(id_=id_,boundary=boundary,**kwargs)
            self.id=id_
            self.points=boundary
        else:
            pass
        
    def contains(self,coord:list):
        if self.left<=coord[0]<=self.right and self.bottom<=coord[1]<=self.top:
            return True
        else:
            return False
    
    def __getnewargs__(self):
        return (self.id,self.points,self.left,self.right,self.bottom,self.top)
    
    def division(self,ndiv=2):
        pass
    
    @property
    def width(self):
        return self.right-self.left
        
    @property
    def height(self):
        return self.top-self.bottom
    
    
class ClusteringBasedPartition(object):
    
    
    def __init__(self):
        pass
    
    
    def road_network_partiton(self,roads:gpd.GeoDataFrame,to_crs=3857):
        ngdf,egdf=ox.graph_to_gdfs(roads)
        ngdf1=ngdf.to_crs(3857)
        pts=[list(pt.coords)[0] for pt in ngdf.geometry.values]
        optics=OPTICS()
        
class PartitionByGrid(object):
    
    
    def __init__(self,region:list=None,distance_method="haversine",**kwargs):
        self.__dict__.update(kwargs)
        if isinstance(region,list):
            self.partition_region(region)
        elif isinstance(region,Polygon):
            self.partition_region(region.points)
        self.distance_method=distance_method
        
    def partition_region(self,bounds,width=100,height=100):
        print("start partition preprocess")
        self.left=bounds[0]
        self.right=bounds[2]
        self.bottom=bounds[1]
        self.top=bounds[3]
        if self.distance_method=="haversine":
            
            x_dist=gps_distance((self.left,self.top),(self.right,self.top))
            y_dist=gps_distance((self.left,self.top),(self.left,self.bottom))
        elif self.distance_method=="euclidean":
            x_dist=self.right-self.left
            y_dist=self.top-self.bottom
            
        self.width_distance=x_dist
        self.height_distance=y_dist
        
        x_times=self.width_distance/float(width)
        y_times=self.height_distance/float(height)
        x_diff=self.right-self.left
        y_diff=self.top-self.bottom
        self.width=x_diff
        self.height=y_diff
        
        self.grid_width_distance=width
        self.grid_height_distance=height
        self.grid_width=x_diff/x_times
        self.grid_height=y_diff/y_times
        
        self.row_count=math.floor(y_times)+1
        self.column_count=math.floor(x_times)+1
        print("end partition preprocess")
        print("total row:{0} column:{1}".format(self.row_count,self.column_count))
        self.grids={}
        
        self.type_="square" if width==height else "grid"
        #generate all grids
#        for row in range(0,self.row_count):
#            
#            for col in range(0,self.column_count):
#                print("generating grid row:{0} column:{1}".format(row,col))
#                row_index=row+1
#                col_index=col+1
#                left=self.left+col*self.grid_width
#                right=left+self.grid_width
#                top=self.top+row*self.grid_height
#                bottom=self.top-self.grid_height
#                points=[(left,top),(right,top),(right,bottom),(left,bottom),(left,top)]
#                grid=Region(id_="{0}-{1}".format(row_index,col_index),points=points,type_=type_)
#                self.grids[grid.id]=grid
        print("completed partition region")
        return self.grids
    
    def is_contain(self,coordinate:list):
        if self.left<=coordinate[0]<=self.right and self.bottom<=coordinate[0]<=self.top:
            return True
        else:
            return False
        
    def get_grid_by_id(self,id_):
        return self.grids.get(id_)
    
    def get_grid_id(self,row,column):
        return "{0},{1}".format(row,column)
    
    def get_grid_by_index(self,row:int,column:int):
        
        id_=self.get_grid_id(int(row),int(column))
        if id_ not in self.grids:
            l=self.left+column*self.grid_width
            b=self.bottom+row*self.grid_height
            r=l+self.grid_width
            t=b+self.grid_height
            self.grids[id_]=GridRegion(id_,(l,b,r,t),type_=self.type_,row=row,column=column)
        return self.get_grid_by_id(id_)
    
    def get_grid_by_num(self,num):
        keys=list(self.grids.keys())
        return self.get_grid_by_id(keys[num])
    
    def mapping_to_index(self,coordinate:list):
        x=coordinate[0]
        y=coordinate[1]
        
        d_x=x-self.left
        d_y=y-self.bottom
        col=math.floor(d_x/self.grid_width)
        row=math.floor(d_y/self.grid_height)
        return row+1,col+1
    
    def mapping_to_grid_id(self,coord:list):
        row,col=self.mapping_to_index(coord)
        return self.get_grid_id(row,col)
    
    def mapping_to_grid(self,coordinate:list):
        row,col=self.mapping_to_index(coordinate)
        grid=self.get_grid_by_index(row,col)
        if grid is None:
            grid=self.generate_grid_by_coord(coordinate)
        return grid
    
    def in_region(self,coord):
        if self.left<=coord[0]<=self.right and self.bottom<=coord[1]<=self.top:
            return True
        else:
            return False
        
    def generate_grid_by_coord(self,coord:list):
        if self.in_region(coord) is False:
            return None
        row,col=self.mapping_to_index(coord)
        
        right=self.left+col*self.grid_width
        left=right-self.grid_width
        top=self.bottom+row*self.grid_height
        bottom=top-self.grid_height
#        points=[(left,top),(right,top),(right,bottom),(left,bottom),(left,top)]
        grid=GridRegion(id_=self.get_grid_id(row,col),bounds=(left,bottom,right,top),type_=self.type_,row=row,column=col)
        self.grids[grid.id]=grid
        return grid
                

    
        