# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:22:09 2020

@author: Administrator
"""
import sys
sys.path.append("/home/boting/project/fleet_rs/")

from ridesharing.strategy import NearestSharing, APART, PDTL
import pandas as pd
import pickle
import osmnx as ox
import networkx as nx
from ridesharing.requests import *
from ridesharing.simulator import RoadsNetworkBasedSimulator
from ridesharing.manager import VehicleManager
from ridesharing.engine import SimulatorEngine
import os
from multiprocessing.dummy import Pool as ThreadPool


path3 = "../data/wuhan_roads.gpkl"
path6 = '../data/wuhan_ods.pkl'
'''
path3="../data/SanFrancisco_roads.pkl"
path6='../data/SanFrancisco_ods.pkl'
'''
g = nx.read_gpickle(path3)


ods = pd.read_pickle(path6)
requests = ods.copy()
timestamp = requests["pick_up_time"]
requests["timestamp"] = timestamp

pool = ThreadPool(4)

#days = ['2013-10-23','2013-10-24', '2013-10-25','2013-10-26','2013-10-27','2013-10-28','2013-10-29','2013-10-30','2013-10-31']
days = ['2008-06-06', '2008-06-07', '2008-06-08', '2008-06-09']

def simulate(day):
    vm = VehicleManager(g, cache_size=20000, name="sf_ridesharing")
    vm.uniformed_initialize_vehicles()
    engine = SimulatorEngine(vm, name="sf_ridesharing")
    simulator = RoadsNetworkBasedSimulator(engine=engine)
    simulator.set_requests(requests)

    path_demand = "../../data/SanFrancisco_demand_demsc.pkl"
    path_regions = "../../data/SanFrancisco_regions.pkl"

    demands = pd.read_pickle(path_demand)
    with open(path_regions, 'rb') as f:
        regions = pickle.load(f)

    #ns = NearestSharing()
    #apart = APART()
    pdtl = PDTL(regions, demands)

    simulator.simulating("sf_ridesharing", [pdtl], dates=[
        day], cache_dir="./ridesharing_results")


pool.map(simulate, days)
pool.close()
pool.join()
