import sys
sys.path.append('/home/mark/Documents/code/drone')
import os
import subprocess
import numpy as np
from sumo.utils.runSumo import runSumo
from sumo.utils.sumo_loop import sumo_loop
import pandas as pd
import tqdm

# example setup below 
setup = {}

setup['mph2mps'] = 0.44704  # Conversion factor from miles per hour to meters per second.

# Simulation parameters:
setup['sigma'] = 0.5
setup['startTime'] = 0  # [s].
setup['endTime'] = 4800  # [s].

# Road parameters:
setup['speedLimit'] = 50 * setup['mph2mps']  # Speed limit on road [m/s].
setup['roadLength'] = 3000  # Length of road [m].
setup['numLanes'] = 4  # Number of lanes on road.
setup['numStops'] = 150  # Number of stops (evenly spaced) along length of road.

# Car parameters:
setup['carLength'] = 4  # Average car length [m].
setup['carSpeed'] = 100 * setup['mph2mps']  # Maximum car speed [m/s].
setup['carAccel'] = 2.87  # Maximum car acceleration rate [m/s] ([1]).
setup['carDecel'] = 4.33  # Maximum car deceleration rate [m/s] ([1]).

# Truck parameters:
setup['truckLength'] = 7.82  # Average delivery truck length [m] ([2]).
setup['truckSpeed'] = 60 * setup['mph2mps']  # Maximum truck speed [m/s].
setup['truckAccel'] = 1.00  # Maximum truck acceleration rate [m/s] ([1]).
setup['truckDecel'] = 0.88  # Maximum truck deceleration rate [m/s] ([1]).
setup['stopTime'] = 60  # Time required for truck stop [s].

# ratios = np.arange(0.01,1.0,0.05)
ratios = np.logspace(np.log10(0.001),np.log10(0.5),100)
print(ratios)

# for i in tqdm.trange(len(ratios)):
#         ratio = ratios[i]
#         sumo_loop(max_flow = 3000,
#                 df = 10,
#                 ratio = ratio,
#                 setup = setup,
#                 out_dir = "sim_3/%d/"%i)

print('next sim')

setup['endTime'] = 3600  # [s].
setup['speedLimit'] = 30 * setup['mph2mps']  # Speed limit on road [m/s].
setup['roadLength'] = 500  # Length of road [m].
setup['numLanes'] = 2  # Number of lanes on road.
setup['numStops'] = 25  # Number of stops (evenly spaced) along length of road.


for i in tqdm.trange(len(ratios)):
        ratio = ratios[i]
        sumo_loop(max_flow = 2000,
                df = 10,
                ratio = ratio,
                setup = setup,
                out_dir = "2_lane/%d/"%i)


# runSumo(flow=4800,
#         ratio = 0.1,
#         setup=setup,
#         filename='outfile',
#         out_dir='output/',
#         use_gui=False)