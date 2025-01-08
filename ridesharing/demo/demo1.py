# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 23:47:00 2020

@author: Administrator
"""


from ridesharing.engine import SimulatorEngine
from ridesharing.manager import VehicleManager
from ridesharing.strategy import NearestSharing,APART

from ridesharing.requests import *

ns=NearestSharing()
apart=APART()
vm=VehicleManager(g)
vhs=vm.uniformed_initialize_vehicles()
engine=SimulatorEngine(vm,ns,cache_dir="D:\\wuhan\\ridesharing\\")
engine.set_requests(ods1)
nn_results=engine.run()
nn_vhs=engine.reinitialize()
engine.set_strategy(apart)
apart_results=engine.run()

