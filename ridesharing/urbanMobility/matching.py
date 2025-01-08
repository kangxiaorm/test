# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:57:33 2019

@author: Administrator
"""
import pandas as pd 
import numpy as np
import numba
from rtree import index

from gisdata.projectsConverter import gcj02_to_wgs84
from gisdata.distance import get_distance_from_coordinate,euclidean_distance

from .functional_region.region_partition import PartitionByGrid

def matching_od_to_aoi(rtree:index.Index,ods:pd.DataFrame,distance=200):
    o_aoi=['']*ods.index.size
    o_poi=['']*ods.index.size
    d_aoi=['']*ods.index.size
    d_poi=['']*ods.index.size
    for n,i in enumerate(ods.index):
        print("finish: {0} %".format(n/ods.index.size*100))
        #od=ods.loc[i]
        o_location=gcj02_to_wgs84(ods.loc[i,'StartX'],ods.loc[i,'StartY'])
        d_location=gcj02_to_wgs84(ods.loc[i,'EndX'],ods.loc[i,'EndY'])
        #===========================================
        r=rtree.nearest(o_location,3,True)
        
        r2=[v for v in r if get_distance_from_coordinate(o_location[0],o_location[1],v.bbox[0],v.bbox[1])<=distance]
        if len(r2)>0:
            d=r2[0].object
            if d['type']=='aoi':
                o_aoi[n]=d['aoi']
                #o_poi[n]=''
            else:
                #o_aoi[n]=''
                o_poi[n]=d['parent']
        else:
            pass
            #o_aoi[n]=''
            #o_poi[n]=''
        
        r=rtree.nearest(d_location,3,True)
        
        r2=[v for v in r if get_distance_from_coordinate(d_location[0],d_location[1],v.bbox[0],v.bbox[1])<=distance]
        if len(r2)>0:
            d=r2[0].object
            if d['type']=='aoi':
                d_aoi[n]=d['aoi']
                #d_poi[n]=''
            else:
                #d_aoi[n]=''
                d_poi[n]=d['parent']
        else:
            pass
            #d_poi[n]=''
            #d_aoi[n]=''
    try:
        ods['origin_aoi']=o_aoi
        ods["origin_poi"]=o_poi
        ods["destination_aoi"]=d_aoi
        ods["destination_poi"]=d_poi
    except Exception as e:
        print(e,"matching ad to aoi")
    return ods,(o_aoi,o_poi,d_aoi,d_poi) 



def matching_od_to_grid(partition:PartitionByGrid,ods:pd.DataFrame,block_size=10000):
#    partition=PartitionByGrid()
#    grids=partition.partition_region(district,grid_width,grid_height)
    
    for g in partition.grids.values():
        g.origins=[]
        g.origin_ods=[]
        g.destinations=[]
        g.destination_ods=[]
        
    for n,i in enumerate(ods.index):
        if n%block_size==0:
            print("process: {0} completed: {1} %".format(i,n/ods.index.size*100))
        od=ods.loc[i].to_dict()
        o_location=gcj02_to_wgs84(ods.loc[i,'StartX'],ods.loc[i,'StartY'])
        d_location=gcj02_to_wgs84(ods.loc[i,'EndX'],ods.loc[i,'EndY'])
        od["origin_coord"]=o_location
        od["destination_coord"]=d_location
        o_id=partition.mapping_to_grid_id(o_location)
        d_id=partition.mapping_to_grid_id(d_location)
        if o_id not in partition.grids:
            try:
                o_grid=partition.generate_grid_by_coord(o_location)
                o_grid.origin_ods=[]
                o_grid.destination_ods=[]
            except Exception as e:
                print(e)
                print("id:{0} coord:{1}".format(o_id,o_location))
                continue
            
        else:
            o_grid=partition.get_grid_by_id(o_id)
        if d_id not in partition.grids:
            try:
                d_grid=partition.generate_grid_by_coord(d_location)
                d_grid.destination_ods=[]
                d_grid.origin_ods=[]
            except Exception as e:
                print(e)
                print("id:{0} coord:{1}".format(d_id,d_location))
                continue
        else:
            d_grid=partition.get_grid_by_id(d_id)
            
        
        o_grid.origin_ods.append(od)
        d_grid.destination_ods.append(od)
        
    
    num=len(partition.grids)
    for n,g in enumerate(partition.grids.values()):
        print("process grid:{0} index:{1} last:{2} complete:{3}".format(g.id,n,num,n/num*100))
        g.origin_ods=pd.DataFrame(g.origin_ods)
        g.destination_ods=pd.DataFrame(g.destination_ods)
        g.origin_count=g.origin_ods.index.size
        g.destination_count=g.destination_ods.index.size
        g.net_flows=g.destination_count-g.origin_count
        
    return partition

            
            