import argparse

import pickle
import numpy as np
import pandas as pd

from optim.solver import *



def main():

    main_seed = np.random.RandomState(10)
    out_path = '/home/mark/Documents/code/drone/figures/temp/'

    # data for all cities goes here
    city_names = ['Sioux Falls','Anaheim','Chicago']
    city_index = {'SiouxFalls':0 ,'Anaheim':1 ,'ChicagoSketch':2}
    n = [24,416,933]
    m = [76,914,2950]
    trips = [360600,104694,1260907]
    centers = [13,13,13]
    city_keys = ['SiouxFalls','Anaheim','ChicagoSketch']
    omega_data = {2:np.array([15.75812964, 0.02109056]), 
            3:np.array([4.26392855, 0.06173418]),
            4:np.array([1.91730372, 0.05962975])}


    # done with setup
    city = ARGS.CITY
    data = {}

    network_file = '/home/mark/Documents/code/drone/tpnt/%s/%s_net.tntp'%(city,city)
    flow_file = '/home/mark/Documents/code/drone/tpnt/%s/%s_flow.tntp'%(city,city)

    net = pd.read_csv(network_file, skiprows=8, sep='\t')
    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed
    net.drop(['~', ';'], axis=1, inplace=True)
    flow = pd.read_csv(flow_file, sep='\t')
    trimmed = [s.strip().lower() for s in flow.columns]
    flow.columns = trimmed
    
    # save data
    data['net'] = net
    data['flows'] = flow
    data['name'] = city_names[city_index[city]]
    data['n'] = n[city_index[city]]
    data['m'] = m[city_index[city]]
    data['trips'] = trips[city_index[city]]
    data['node_map'] = lambda node: (node < 13)*node + (node == 13)*0 + (node > 13)*(node-1)


    # lets look at the volume on the road and choose cutoff for lanes
    lanes = []
    for vol in data['net']['capacity']:
        if vol > data['net']['capacity'].median():
            lanes.append(3)
        else: lanes.append(2)
    data['edge_lanes'] = lanes




    net_params = {}
    optim_params = {}

    # pre load for syntax 
    n = data['n']
    m = data['m']
    net = data['net']
    edge_lanes = data['edge_lanes']
    flows = data['flows']
    node_map = data['node_map']
    # adjecency matrix 
    A = np.zeros((n,n),dtype=int)
    # edge capacity equal to flow of solution * scale
    capacity = np.zeros((n,n),dtype=int)
    # lanes (2,3) (divide in half)
    lanes = np.zeros((n,n),dtype=int)
    # latency on edges
    l_road = np.zeros((n,n),dtype=int)
    # car flows
    flow_c = np.zeros((n,n),dtype=int)

    for o, d, vol, length, e in zip(net['init_node'],net['term_node'],net['capacity'],net['length'], net.index):
        # check if center node, and shift to first row/column
        o,d = node_map(o),node_map(d)
        A[o,d] = 1
        l_road[o,d] = length # arbitrary to convert, we will compare precentages
        capacity[o,d] = vol # 
        lanes[o,d] = edge_lanes[e]
    
    if city == 'ChicagoSketch': #double flow for chicago
        for o, d, flow in zip(flows['from'],flows['to'],flows['volume']):
            flow_c[(node_map(o),node_map(d))] = flow*2
    else:
        for o, d, flow in zip(flows['from'],flows['to'],flows['volume']):
            flow_c[(node_map(o),node_map(d))] = flow


    demand = data['trips']/72.12 #this gets 5000 demand for sioux falls
    net_params['A'] = A
    net_params['flow_c'] = flow_c
    net_params['capacity'] = capacity
    net_params['lanes'] = lanes
    net_params['l_road'] = l_road
    net_params['l_drone'] = None # for now
    net_params['omega'] = omega_data
    net_params['cutoff'] = ARGS.CUTOFF
    net_params['Beta'] = data['trips']
    net_params['D_v'] = np.ones(n-1,dtype=int)*demand
    # net_params['delivery_time'] n-1 0
    net_params['pos'] = None # we will not render

    #### OPTIM PARAMS ####
    optim_params = {}
    optim_params['p_per_truck'] = 125 #125
    optim_params['c_t'] = 30
    optim_params['c_d'] = 0.5
    # assumes trucks are cheaper 
    c_min = optim_params['c_t']*demand*(n-1)/optim_params['p_per_truck']
    optim_params['C_0'] = int(c_min*1.44928)

    optim_params['MIP_gap'] = 0.00001
    optim_params['allow_drones'] = True

    solution = () # tuple for easy data coupling
    print('-----'*8)
    print(city)
    print('-----'*8)
    data['l_drone'] = []
    my_solver = net_solver(net_params)
    # first we need to get drone latency by scaling shortest path
    paths = create_paths(my_solver.G, cutoff=1) # just the shortest
    for path in paths:
        latency = 0
        for i in range(len(path)-1):
            edge = tuple(path[i:i+2])
            e = my_solver.edge_to_idx(edge)
            latency += my_solver.l_road[e] + np.dot([my_solver.omega[0][e],my_solver.omega[1][e]],[0,my_solver.flow_c[e]]) 
        data['l_drone'].append(latency*0.8) # scale the latency
    # input latency after
    my_solver.l_drone = data['l_drone']
    # save params
    #### OPTIMIZE ####
    for gamma in [1,0.5,0]:
        optim_params['Gamma'] = gamma
        sol = my_solver.optimize(optim_params)
        solution += ((gamma,sol),)

    pickle.dump(solution,open('/home/mark/Documents/code/drone/NC_results/'+ARGS.CITY+'_'+str(ARGS.CUTOFF)+'.p','wb'))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # these three args are irrelevant for now
    parser.add_argument('--CUTOFF', type=int, help="num of paths per od pair")
    parser.add_argument('--CITY', type=str,  help="name of city dir")
    ARGS = parser.parse_args()
    print(ARGS)
    
    main()