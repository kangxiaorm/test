import sys
sys.path.append("/home/boting/project/fleet_rs/")

from ridesharing.engine import SimulatorEngine
from ridesharing.manager import VehicleManager
from ridesharing.strategy import NearestSharing,APART,PDTL
from ridesharing.simulator import RoadsNetworkBasedSimulator
from ridesharing.requests import *
import networkx as nx
import pandas as pd
import osmnx as ox
import pickle

sf_od = "../data/SanFrancisco_ods.pkl"
with open(sf_od, 'rb') as v:
    data2 = pickle.load(v)

wh_demand = "../data/wuhan_regions_n.pkl"
sf_demand = "../data/SanFrancisco_regions.pkl"

with open(sf_demand, 'rb') as v:
    data2 = pickle.load(v)

with open(wh_demand, 'rb') as f:
    data = pickle.load(f)



data