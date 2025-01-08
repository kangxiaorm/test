# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:41:02 2020

@author: Administrator
"""
import pandas as pd 


def demand(ods:pd.DataFrame,n,partition,unit=1,path=None):
    dates:pd.Series=ods["StartDate"]
    dates=dates.unique()
    demands={i:{} for i in range(int(24*60/unit))}
    p=0
    for i,od in ods.iterrows():
        pt=(od.StartX,od.StartY)
        if partition.in_region(pt):
            
            id_=partition.mapping_to_grid_id((od.StartX,od.StartY))
            tid=get_time_id(od.StartTime,unit=unit)
            if id_ not in demands[tid]:
                demands[tid][id_]=0
            demands[tid][id_]+=1
            
        p1=int(i/n*100)
        if p1!=p:
            p=p1
            print("complete:{0} % od_id:{1}".format(p,i))
    r=pd.DataFrame(demands).T
    path="G:\\demands({0}).csv".format(unit) if path is None else path
    try:
        r.to_csv(path)
    except:
        pass
    
    return r

def demand2(ods:pd.DataFrame,n,partition,unit=1,path=None):
    dates:pd.Series=ods["StartDate"]
    dates=dates.unique()
    demands={}
    for date in dates:
        dods=ods[ods.StartDate==date]
        print("processing date:{0}".format(date))
        for i,od in dods.iterrows():
            pt=(od.StartX,od.StartY)
            if partition.in_region(pt):
#                id_=partition.mapping_to_grid_id(pt)
                tid=get_time_id(od.StartTime,unit=unit)
                k=(date,tid)
                if k not in demands:
                    demands[k]=0
                demands[k]+=1
    
    s=pd.Series(demands)
    
    return s
    
def get_time_id(time: str, unit: int = 30):
    time_id = int((pd.Timestamp(time)-pd.Timestamp("00:00:00")
                   ).total_seconds()/60/unit)
    return time_id

def recoding_time_id(id_,unit:int=1):
    ts=pd.Timestamp("00:00:00")
    dt=pd.Timedelta(minutes=id_)
    t=ts+dt
    return t.strftime("%H:%M:%S")
    
def demand1(ods:pd.DataFrame,n,partition,unit=30,path=None):
    dates:pd.Series=ods["StartDate"]
    dates=dates.unique()
    demands={}
    for date in dates:
        print("processing date:{0}".format(date))
        dods=ods[ods.StartDate==date]
       
        for i,od in dods.iterrows():
            pt=(od.StartX,od.StartY)
            if partition.in_region(pt):
                id_=partition.mapping_to_grid_id(pt)
                tid=get_time_id(od.StartTime,unit=unit)
                k=(date,tid)
                if k not in demands:
                    demands[k]={}
                if id_ not in demands[k]:
                    demands[k][id_]=0
                demands[k][id_]+=1
    r=pd.DataFrame(demands).T
    path="G:\\demands({0}).csv".format(unit) if path is None else path
    try:
        r.to_pickle(path)
    except:
        pass
    
    return r 