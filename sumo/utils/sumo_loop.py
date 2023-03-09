import sys
sys.path.append('/home/mark/Documents/code/drone')
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sumo.utils.runSumo import runSumo

def sumo_loop(max_flow, df, ratio, setup, out_dir):
    '''
    Incremently increases the total flow while keeping 
    ratio constant. Plots the corresponding avg flow.

    Inputs:
        max_flow - maximum flow the simulation runs until
        df       - flow differentials we increment by 
        ratio    - the ratio to keep during loop
        setup    - network setup
        out_dir  - where sim results are kept
        filename - name of pickle and plot associated
    '''
    # create required dirs
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        # print("DIRECTORY ALREADY EXISTS")
        return

    # we want to collect car and truck latency
    # l_car = []
    # l_truck = []
    # avg_flows = []
    Data = {}
    flows = np.arange(df, max_flow, df)
    Data['flows'] = flows
    for flow in flows:
        # run and extract
        inner_dir = out_dir+str(flow)+"/"
        os.mkdir(inner_dir)
        runSumo(flow, ratio, setup, filename='outfile', out_dir=inner_dir)


