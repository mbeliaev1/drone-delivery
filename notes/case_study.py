import os
import sys
sys.path.append('/home/mark/Documents/code/drone')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from geopy.distance import lonlat, distance

from optim.solver import *
# from convex_solver import *
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

FIGH = 2.8

main_seed = np.random.RandomState(10)
out_path = '/home/mark/Documents/code/drone/figures/temp/'

# this loads in the appropriate data
city = 'SiouxFalls'
# num_nodes = 1052
# coordinate_available = False

network_file = '/home/mark/Documents/code/drone/tpnt/%s/%s_net.tntp'%(city,city)
coordinate_file = '/home/mark/Documents/code/drone/tpnt/%s/%s_node.tntp'%(city,city)
flow_file = '/home/mark/Documents/code/drone/tpnt/%s/%s_flow.tntp'%(city,city)

net = pd.read_csv(network_file, skiprows=8, sep='\t')
trimmed = [s.strip().lower() for s in net.columns]
net.columns = trimmed
net.drop(['~', ';'], axis=1, inplace=True)

flows = pd.read_csv(flow_file, sep='\t')
trimmed = [s.strip().lower() for s in flows.columns]
flows.columns = trimmed

coordinates = pd.read_csv(coordinate_file, sep='\t')
trimmed = [s.strip().lower() for s in coordinates.columns]
coordinates.columns = trimmed


CNODE = 13
CUTOFF = 50

# drone_speed = (25/60)*(1/np.sqrt(2)) #km/minute
drone_speed = 25/60
node_map = lambda node: (node < CNODE)*node + (node == CNODE)*0 + (node > CNODE)*(node-1)
n = len(coordinates.index)
m = len(net.index)
cap_cutoffs = [12000,18000]


# imports lane data, all 3 lane roads are 50 mph highways 
temp = [1, 3, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 1, 1, 3, 2, 1, 2, 3, 3, 2, 1, 2, 1, 2, 2, 2, 2, 2, 3, 1, 1, 3, 3, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 1, 3, 3, 2, 2, 2, 2, 1]

edge_lanes = []
for lanes in temp:
    if lanes == 3:
        edge_lanes.append(3)
    else:
        edge_lanes.append(2)


# adjecency matrix 
A = np.zeros((n,n),dtype=int)
# edge capacity
capacity = np.zeros((n,n),dtype=int)
# lanes (2,3,4)
lanes = np.zeros((n,n),dtype=int)
# latency on edges
l_road = np.zeros((n,n),dtype=int)


for o, d, cap, length, e in zip(net['init_node'],net['term_node'],net['capacity'],net['length'], net.index):
    # check if center node, and shift to first row/column
    o,d = node_map(o),node_map(d)
    A[o,d] = 1
    l_road[o,d] = (length*100)/60 # convert to minutes
    capacity[o,d] = cap
    lanes[o,d] = edge_lanes[e]
    # if cap > cap_cutoffs[1]:
    #     lanes[o,d] = 4
    # elif cap < cap_cutoffs[0]:
    #     lanes[o,d] = 2
    # else:
    #     lanes[o,d] = 3

# car flow along edges
flow_c = np.zeros((n,n),dtype=int)

for o, d, flow in zip(flows['from'],flows['to'],flows['volume']):
    flow_c[(node_map(o),node_map(d))] = flow

# latency of drone paths
l_drone = np.zeros(n-1,dtype=int)
pos = np.zeros((n,2))

for node, x, y in zip(coordinates['node'],coordinates['x'],coordinates['y']):
    # nodes start at 1
    pos[node_map(node)] = np.array([x,y])


# drone graph latencies
for node in range(1,n):
    city_x= pos[0] # origin is always center node
    city_y=	pos[node] 
    D = distance(lonlat(*city_x),lonlat(*city_y)).km
    l_drone[node-1] = D/drone_speed# l_drone length is n-1


# node_shift = {8:[-.006,.003],9:[.003,-.006],15:[-.006,.003],16:[.003,-.006]}
left_squeeze = 0.015
node_shift = {0:[left_squeeze+0.01,0.007],
              1:[left_squeeze+0.007,-0.012],
              2:[0,-0.005],
              3:[left_squeeze,0.013],
              4:[0,0.018],
              5:[0,0.018],
            #   6:[0,0.003],
              7:[0,0.007],
              8:[0,0.008],
              9:[0,0.015],
            #   10:[0,0.003],
              # 11:[-0.005,0],
              12:[left_squeeze,0],
            #   13:[0,0.003],
            #   14:[0,.002],
              15:[0,.007],
              16:[0,-.002],
            #   17:[0,.001],
              18:[0,-.006],
              19:[0,-.011],
              20:[0,-.012],
            #   21:[0,.005],
            #   22:[0,.001],
              23:[0,-.005]}
# redo position but shifted
for node, x, y in zip(coordinates['node'],coordinates['x'],coordinates['y']):
    if node_map(node) in node_shift.keys():
        pos[node_map(node)] = np.array([x,y]) + node_shift[node_map(node)]

net_params = {}
omega = {2:np.array([15.75812964, 0.02109056]), 
         3:np.array([4.26392855, 0.06173418]),
         4:np.array([1.91730372, 0.05962975])}

demand = 5000
net_params['A'] = A
net_params['flow_c'] = flow_c
net_params['capacity'] = capacity
net_params['lanes'] = lanes
net_params['l_road'] = l_road
net_params['l_drone'] = l_drone
net_params['omega'] = omega  
net_params['cutoff'] = CUTOFF
net_params['Beta'] = 360600
net_params['D_v'] = np.ones_like(l_drone,dtype=int)*demand
# net_params['delivery_time'] = 0
G = nx.DiGraph(A)

net_params['pos'] = pos

fig_params = {}

fig_params['ORANGE'] = '#FF9132'
fig_params['TEAL'] = '#0598B0'
fig_params['GREEN'] = '#008F00'
fig_params['PURPLE'] = '#8A2BE2'
fig_params['GRAY'] = '#969696'

fig_params['N_COLOR'] = [fig_params['GREEN']]+[fig_params['ORANGE']]*len(net_params['l_drone']) 
fig_params['N_SIZE'] = 100
fig_params['N_FSIZE'] = 8
fig_params['E_FSIZE'] = 8
fig_params['E_WIDTH'] = 0.75
fig_params['E_ALPHA'] = 0.9
fig_params['E_COLOR'] = fig_params['GRAY']
fig_params['ARROW_SIZE'] = 5
# fig_params['FIG_W'] = 4
# fig_params['FIG_H'] = 4
fig_params['arc_rad'] = 0.1

fig_params['rc_params'] = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "lines.linewidth": 2}

fig_params['rc_params']["text.latex.preamble"] = (r'\usepackage{amsmath,amsthm,amssymb}')
fig_params['rc_params']["text.latex.preamble"] += (r'\usepackage{mathtools}')
fig_params['rc_params']["text.latex.preamble"] += (r'\usepackage{tabulary}')
fig_params['rc_params']["text.latex.preamble"] += (r'\usepackage{booktabs}')
optim_params = {}
# set random variables for now
         

optim_params['p_per_truck'] = 125 #125
optim_params['C_0'] = 40000
optim_params['c_t'] = 30
optim_params['c_d'] = 0.5

optim_params['MIP_gap'] = 0.00001


FIG_SAVEPATH = '/home/mark/Documents/code/drone/optim/results/Sioux_setup.pdf'
my_solver = net_solver(net_params)

# INDIVIDUAL SAVED FIGURE PARAMS GO HERE
# SETUP
# main_w = 0.8
cbar_w = 0.99
cbar_h = 0.025


fig_params['setup'] = {
        'suptitle': 'Sioux Falls Transportation Network Flows',
        'widths': [1-cbar_w/2,cbar_w,1-cbar_w/2],
        'heights': [1-cbar_h,cbar_h],
        'l':0,
        'r':1,
        't':1,
        'b':0.06,
        'wspace':0,
        'hspace':0,
        'figw':3,
        'figh':FIGH,
        'cmin':0,
        'cmax':300,
        'cticks':[0,100,200,300],
        'cticklabels':['$0 \\%$','$100 \\%$','$200 \\%$','$300\\%$'],
        'clabel':'Edges: Car Flow ($\\%$ of $f^0$)',
        'ctrunc':[0.2,1]}

my_solver.visualize_setup(fig_params, FIG_SAVEPATH)


optim_params['Gamma'] = 1
optim_params['allow_drones'] = False 

left_sol = my_solver.optimize(optim_params)

optim_params['Gamma'] = 0
optim_params['allow_drones'] = False 

right_sol = my_solver.optimize(optim_params)

FIG_SAVEPATH = '/home/mark/Documents/code/drone/optim/results/Sioux_nodrones.pdf'
sol_w  = 0.4

top_h = 0.001
cbar_h = 0.035
res_h = 1-top_h-2*cbar_h
# space = (1-top_h-res_h-cbar_h-cbar_h)/2

fig_params['setup'] = {
        'suptitle': 'Optimal Routing Strategies without Drones',
        'widths': [sol_w,0.5-sol_w,0.5-sol_w,sol_w],
        'heights': [top_h,res_h,cbar_h,cbar_h],
        'l':0,
        'r':1,
        't':1,
        'b':0.06,
        'wspace':0,
        'hspace':0.6,
        'figw':7,
        'figh':FIGH,
        'cmin':100,
        'cmax':106,
        'cticks':[100,102,104,106],
        'cticklabels':['$100 \\%$','$102 \\%$','$104 \\%$','$106\\%$'],
        'clabel':'Edges: Road Latency ($\\%$ of $\\ell^C$)',
        'ctrunc':[0.2,0.5]}

fig_params['E_WIDTH'] = 1.5
fig_params['E_ALPHA'] = 0.9


my_solver.visualize_sol(left_sol, right_sol, fig_params, FIG_SAVEPATH)

optim_params['Gamma'] = 1
optim_params['allow_drones'] = True 

left_sol = my_solver.optimize(optim_params)

optim_params['Gamma'] = 0
optim_params['allow_drones'] = True 

right_sol = my_solver.optimize(optim_params)

FIG_SAVEPATH = '/home/mark/Documents/code/drone/optim/results/Sioux_drones.pdf'
sol_w  = 0.4

top_h = 0.001
cbar_h = 0.035
res_h = 1-top_h-2*cbar_h
# space = (1-top_h-res_h-cbar_h-cbar_h)/2

fig_params['setup'] = {
        'suptitle': 'Optimal Routing Strategies with Drones',
        'widths': [sol_w,0.5-sol_w,0.5-sol_w,sol_w],
        'heights': [top_h,res_h,cbar_h,cbar_h],
        'l':0,
        'r':1,
        't':1,
        'b':0.06,
        'wspace':0,
        'hspace':0.6,
        'figw':7,
        'figh':FIGH,
        'cmin':100,
        'cmax':106,
        'cticks':[100,102,104,106],
        'cticklabels':['$100 \\%$','$102 \\%$','$104 \\%$','$106\\%$'],
        'clabel':'Edges: Road Latency ($\\%$ of $\\ell^C$)',
        'ctrunc':[0.2,0.5]}

fig_params['E_WIDTH'] = 1.5
fig_params['E_ALPHA'] = 0.9


my_solver.visualize_sol(left_sol, right_sol, fig_params, FIG_SAVEPATH)

my_solver.l_drone = np.sqrt(2)*np.asarray(my_solver.l_drone, dtype='float64')

optim_params['Gamma'] = 1
optim_params['allow_drones'] = True 

left_sol = my_solver.optimize(optim_params)

optim_params['Gamma'] = 0
optim_params['allow_drones'] = True 

right_sol = my_solver.optimize(optim_params)

FIG_SAVEPATH = '/home/mark/Documents/code/drone/optim/results/Sioux_slow_drones.pdf'
sol_w  = 0.4

top_h = 0.001
cbar_h = 0.035
res_h = 1-top_h-2*cbar_h
# space = (1-top_h-res_h-cbar_h-cbar_h)/2

fig_params['setup'] = {
        'suptitle': 'Optimal Routing Strategies with Slower Drones',
        'widths': [sol_w,0.5-sol_w,0.5-sol_w,sol_w],
        'heights': [top_h,res_h,cbar_h,cbar_h],
        'l':0,
        'r':1,
        't':1,
        'b':0.06,
        'wspace':0,
        'hspace':0.6,
        'figw':7,
        'figh':FIGH,
        'cmin':100,
        'cmax':106,
        'cticks':[100,102,104,106],
        'cticklabels':['$100 \\%$','$102 \\%$','$104 \\%$','$106\\%$'],
        'clabel':'Edges: Road Latency ($\\%$ of $\\ell^C$)',
        'ctrunc':[0.2,0.5]}

fig_params['E_WIDTH'] = 1.5
fig_params['E_ALPHA'] = 0.9


my_solver.visualize_sol(left_sol, right_sol, fig_params, FIG_SAVEPATH)