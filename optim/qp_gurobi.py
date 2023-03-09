# import os
# import sys
# import pickle
# import latex
# import matplotlib.font_manager
from networkx.generators.small import cubical_graph
import numpy as np
from numpy import linalg as la
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import itertools
import gurobipy as gp
from gurobipy import GRB


# NODE_SIZE = 150
# NODE_FSIZE = 8
# EDGE_FSIZE = 8
# EDGE_WIDTH = 3
# EDGE_ALPHA = 0.9
# EDGE_COLOR = GRAY
# ARROW_SIZE = 10
# FIG_WIDTH = 12
# FIG_HEIGHT = 12
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "lines.linewidth": 2
})
cmap = plt.cm.Wistia

def convert_to_e(G,omega,flow_c,lanes,l_road,nom_flow):
    '''
    Converts dictionairy, flow matrix, and lane matrix
    to be compatible with outedge lists from G
    '''
    omega_0 = []
    omega_1 = []
    flow_ec = []
    lanes_e = []
    l_road_e = []

    for edge in G.edges:
        v = edge[0]
        w = edge[1]
        n_lanes = lanes[v][w]
        
        lanes_e.append(n_lanes)
        flow_ec.append(flow_c[v][w])
        l_road_e.append(l_road[v][w])

        omega_0.append(omega[n_lanes][0]*l_road[v][w]/nom_flow[v][w])
        omega_1.append(omega[n_lanes][1]*l_road[v][w]/nom_flow[v][w])

    return np.asarray([omega_0,omega_1]), np.asarray(flow_ec), np.asarray(lanes_e), np.asarray(l_road_e)

def create_B(m,paths,edge_to_idx):
    '''
    Creates matrix that transforms from paths to edges
    Output shapes: m x p 
    '''
    B = np.zeros((m,len(paths)),dtype=int)
    for i_path in range(len(paths)):
        path = paths[i_path]
        for i in range(len(path)-1):
            edge = tuple(path[i:i+2])
            B[edge_to_idx(edge)][i_path] = 1
    return B

def create_C(n,paths):
    '''
    Creates matrix that transforms paths to their arrival nodes
    Output shapes: n x p 
    '''
    C = np.zeros((n-1,len(paths)),dtype=int)
    for i_path in range(len(paths)):
        end_node = paths[i_path][-1]
        C[end_node-1][i_path] = 1
        
    return C

def create_D(G,edge_to_idx):
    '''
    Creates matrix that nodes to their downlink edges (arriving)
    Output shapes: m x v 
    '''
    D = np.zeros((G.number_of_edges(),G.number_of_nodes()-1),dtype=int)
    for node in range(G.number_of_nodes()-1):
        for edge in G.in_edges(node+1):
            D[edge_to_idx(edge)][node] = 1
    return D

# def create_O(G,edge_to_idx):
#     '''
#     Creates matrix that nodes to their downlink edges (arriving)
#     Output shapes: m x v 
#     '''
#     O = np.zeros((G.number_of_nodes()-1,G.number_of_edges()),dtype=int)
#     for node in range(G.number_of_nodes()-1):
#         for in_edge, out_edge in zip(G.in_edges(node+1),G.out_edges(node+1)):
#             O[node][edge_to_idx(in_edge)] = 1
#             O[node][edge_to_idx(out_edge)] = -1
#     return O

def solve_qp(G,paths,omega,lanes,nom_flow,flow_c,l_road,l_drone,D_v,p_per_truck,C_0,c_t,c_d,Gamma,Beta):
    '''
    Inputs:
        G         - Networkx graph object| directional graph, connected)
        paths     - tuple len p| Set of Paths that can be used (tuple containing integer 
                    lists of varying length)
        omega     - Dict| Omega vector required to estimate latency. 
                        Key   : Value
                        2 : 2-lane road weights [w_0,w_1s]
                        3 : same for 3 lane road
                        4 : ...
        lanes     - dim: nxn| Integer defining num of lanes along edges
        flow_ec   - dim: nxn| Nominal car flow along edges 
        l_drone   - dim: v-1| Vector representing latency for drone paths
        D_v       - dim: v-1| Vector defining parcel demands for each vertex
        m         - scalar| Parcels Per Truck
        C_0       - scalar| maximum operational cost (price contraint scalar)
        c_t       - scalar| hourly operational cost per truck  
        c_d       - scalar| hourly operational cost per drone
        Gamma     - scalar| tradeoff in cost function
    '''
    # get info from graph
    n = G.number_of_nodes()
    m = G.number_of_edges()
    edge_to_idx = lambda edge: list(G.edges).index(edge)
    omega, flow_c, lanes, l_road = convert_to_e(G,omega,flow_c,lanes,l_road,nom_flow)
    # get matrices and constants
    matB = create_B(m, paths, edge_to_idx)
    matC = create_C(n, paths)
    matD = create_D(G, edge_to_idx)
    matE = np.matmul(matD,matC)  
    D_tot = np.sum(D_v)
    # create main obj params
    obj_Q = (p_per_truck*Gamma/D_tot)*(matB.T@np.diag(omega[0])@matE + matB.T@np.diag(omega[1])@matB)
    obj_a = (p_per_truck*Gamma/D_tot)*(matB.T@np.diag(omega[1])@flow_c - matC.T@l_drone)
    obj_a += ((1-Gamma)/Beta)*(flow_c@np.diag(omega[0])@matE + flow_c@np.diag(omega[1])@matB)
    offset = (Gamma/D_tot)*(np.inner(l_drone,D_v)) + ((1-Gamma)/Beta)*np.inner(flow_c,l_road)
    # cost constraint params
    cost_cons = matC.T@np.ones(n-1)
    cost_ub = (C_0-c_d*D_tot)/(c_t-c_d*p_per_truck)
    # print(cost_ub)
    # demand constraint
    demand_cons = p_per_truck*matC
    demand_ub = D_v
    # create gurobi model
    solver = gp.Model()
    # intialize solver either GRB.INTEGER for MIQP or GRB.CONTINUOUS for QP 
    x = solver.addMVar(len(paths), lb=0, vtype=GRB.INTEGER)
    solver.setObjective(x @ obj_Q @ x + obj_a@x)
    if c_t >= c_d*p_per_truck:
        solver.addConstr(cost_cons@x <= cost_ub)
    else:
        solver.addConstr(cost_cons@x >= cost_ub) 
    solver.addConstr(demand_cons@x <= demand_ub) 
    # solver.setObjCon(offset)
    
    return solver, cost_cons, cost_ub


def convert_sol(X,G,paths,D_v,p_per_truck):
    '''r
    Inputs:
        X           - dim: p| sol list of length p, representing flow along paths
        G           - Networkx graph object| directional graph, connected)
        paths       - tuple len p| Set of Paths that can be used (tuple containing integer 
        D_v         - dim: v-1| Vector defining parcel demands for each vertex
        p_per_truck - scalar| Parcels Per Truck
    '''
    # get info from graph
    n = G.number_of_nodes()
    m = G.number_of_edges()
    edge_to_idx = lambda edge: list(G.edges).index(edge)
    # get matrices and constants
    matB = create_B(m, paths, edge_to_idx)
    matC = create_C(n, paths)
    matD = create_D(G, edge_to_idx)
    matE = np.matmul(matD,matC)  
    # flow of trucks along edges
    flow_t = matB@X
    # flow of stoppign trukcs along edges
    flow_st = matE@X
    # demand satisfied by trucks at ea node
    demand_t = (p_per_truck)*matC@X
    # flow of drones along their aerial paths to ea node
    flow_d = D_v - demand_t 

    return flow_t, flow_st, flow_d, demand_t

def visualize(G,paths,omega,lanes,nom_flow,flow_c,l_road,l_drone,D_v,p_per_truck,C_0,c_t,c_d,Gamma,Beta,title,save_path,X =None, pos = None, plot_toy = False):
    '''
    Inputs:
        G         - Networkx graph object| directional graph, connected)
        paths     - tuple len p| Set of Paths that can be used (tuple containing integer 
                    lists of varying length)
        omega     - Dict| Omega vector required to estimate latency. 
                        Key   : Value
                        2 : 2-lane road weights [w_0,w_1s]
                        3 : same for 3 lane road
                        4 : ...
        flow_ec   - dim: m| Nominal car flow along edges 
        l_drone   - dim: v-1| Vector representing latency for drone paths
        D_v       - dim: v-1| Vector defining parcel demands for each vertex
        m         - scalar| Parcels Per Truck
        C_0       - scalar| maximum operational cost (price contraint scalar)
        c_t       - scalar| hourly operational cost per truck  
        c_d       - scalar| hourly operational cost per drone
        Gamma     - scalar| tradeoff in cost function
        X         - dim: p| sol list of length p, representing flow along paths
        pos       - position format of the vertices for newtorkx        
    '''
    n = G.number_of_nodes()
    m = G.number_of_edges()
    edge_to_idx = lambda edge: list(G.edges).index(edge)
    # get matrices and constants
    matB = create_B(m, paths, edge_to_idx)
    matC = create_C(n, paths)
    D_tot = np.sum(D_v)
    # convert to e format
    omega, flow_c, lanes, l_road = convert_to_e(G,omega,flow_c,lanes,l_road,nom_flow)

    # check for pos
    if pos is None:
        pos = nx.planar_layout(G)
    # check for solution
    sol_available = False
    if X is not None:
        flow_t, flow_st, flow_d, demand_t = convert_sol(X,G,paths,D_v,p_per_truck)
        sol_available = True
        print(flow_t, flow_st, flow_d, demand_t)

    # setup the plotting environment
    NODE_COLOR = [GREEN]+[ORANGE]*(n-1) 
    NODE_SIZE = 100
    NODE_FSIZE = 8
    EDGE_FSIZE = 6
    EDGE_WIDTH = 2
    EDGE_ALPHA = 0.9
    EDGE_COLOR = GRAY
    ARROW_SIZE = 10
    FIG_WIDTH = 3.2
    FIG_HEIGHT = 3.2
    # NODE_SIZE = 150
    # NODE_FSIZE = 8
    # EDGE_FSIZE = 8
    # EDGE_WIDTH = 3
    # EDGE_ALPHA = 0.9
    # EDGE_COLOR = GRAY
    # ARROW_SIZE = 10
    # FIG_WIDTH = 12
    # FIG_HEIGHT = 12
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "lines.linewidth": 2
    })
    cmap = plt.cm.Wistia

    # create fig
    fig = plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT))
    widths = [0.9,0.0125,0.05,0.0125]
    heights = [0.5,0.5]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths,
                            height_ratios=heights,left=-0.03,right=0.88,top=0.95,bottom=0.04)
    spec.update(wspace=0, hspace=0.1)
    ax_drone = fig.add_subplot(spec[0,0])
    ax_truck = fig.add_subplot(spec[1,0])
    ax_color = fig.add_subplot(spec[1,2])
    ax_res = fig.add_subplot(spec[0,1:4])
    # turn them off
    ax_drone.axis('off')
    ax_truck.axis('off')
    ax_res.axis('off')
    # set labels and titles
    ax_drone.set_title('Aerial Network',pad=0)
    ax_truck.set_title('Road Network',pad=0)
    # NETWORK X #
    # for drone net, 0 central node connected to all others
    G_D = nx.DiGraph(np.concatenate(
            ([0],np.ones(n-1),np.zeros(n*n-n))).reshape(n,n))
    # Create node labels 
    n_labels = {}
    for node in pos.keys():
        n_labels[node] = '$v_{%d}$'%node

    if not sol_available:
        ##---------------DRONES-----------------##
        # edge labels
        e_labels = {}
        first_label = True
        for edge in G_D.edges:
            if edge[0] == 0:
                if first_label:
                    e_labels[edge] = 'Demand: $%d$ $\\frac{parcels}{hour}$'%D_v[edge[1]-1]
                    first_label = False
                else:
                    e_labels[edge] = '$%d$'%D_v[edge[1]-1]
        nodes = nx.draw_networkx_nodes(
            G = G_D, 
            pos = pos,
            ax = ax_drone,
            node_size = NODE_SIZE,
            node_color=NODE_COLOR,
        )
        edges = nx.draw_networkx_edges(
            G_D,
            pos,
            ax = ax_drone,
            node_size=NODE_SIZE,
            arrowstyle="->",
            arrowsize=ARROW_SIZE,
            width = EDGE_WIDTH,
            edge_color = EDGE_COLOR
        )
        labels = nx.draw_networkx_labels(
            G_D,
            pos,
            ax = ax_drone,
            labels=n_labels,
            font_size=NODE_FSIZE
        )
        edge_labels = nx.draw_networkx_edge_labels(
            G_D,
            pos,
            ax = ax_drone,
            font_size = EDGE_FSIZE,
            # label_pos = 0.5,
            edge_labels=e_labels,
            verticalalignment='center',
            horizontalalignment='center'
        )

        ##---------------TRUCKS-----------------##
        # edge labels, widths, and colors
        e_labels = {}
        e_widths = []
        e_colors = []
        first_label = True
        avg_l = 0

        for edge in G.edges:
            # labels
            e = edge_to_idx(edge)
            if first_label:
                e_labels[edge] = 'Car Flow: $%d$ $\\frac{cars}{hour}$'%flow_c[e]
                first_label = False
            else:
                e_labels[edge] = '$%d$'%flow_c[e]
            # widths and colors depend on num lanes
            e_widths.append(EDGE_WIDTH*lanes[e])
            l = l_road[e] + np.dot([omega[0][e],omega[1][e]],[0,flow_c[e]]) 
            e_colors.append(l)
            avg_l += l*flow_c[e]
        avg_l /= Beta
        e_colors = np.asarray(e_colors)
        # print('colors: ',e_colors)

        nodes = nx.draw_networkx_nodes(
            G = G, 
            pos = pos,
            ax = ax_truck,
            node_size = NODE_SIZE,
            node_color=NODE_COLOR,
        )
        edges = nx.draw_networkx_edges(
            G,
            pos,
            ax = ax_truck,
            node_size=NODE_SIZE,
            arrowstyle="->",
            arrowsize=ARROW_SIZE,
            width = e_widths,
            edge_color = e_colors,
            edge_cmap= cmap
        )
        labels = nx.draw_networkx_labels(
            G,
            pos,
            ax = ax_truck,
            labels=n_labels,
            font_size=NODE_FSIZE
        )
        edge_labels = nx.draw_networkx_edge_labels(
            G,
            pos,
            ax = ax_truck,
            font_size = EDGE_FSIZE,
            # label_pos = 0.5,
            edge_labels=e_labels,
            verticalalignment='center',
            horizontalalignment='center'
        )
        # COLOR BAR
        pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        pc.set_array(e_colors)
        pc.set_clim(0,4)
        cb = plt.colorbar(pc,cax=ax_color)
        cb.outline.set_visible(False)
        cb.set_ticks([0,1,2,3,4])
        cb.set_ticklabels([0,1,2,3,4])
        ax_color.set_ylabel('Latency without trucks $(minutes)$')
        # RESULT BAR
        anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                        va='center', ha='center')
        # create Latex like string for result
        # res_str = "\\begin{eqnarray*}"
        # res_str += '\\textrm{Societal Latency} &:& %.2f \\textrm{ minutes}'%avg_l + '\\\\'
        # res_str += '\\textrm{Max Cost} &:& %d \\textrm{ }\\frac{\\textrm{dollars}}{\\textrm{hour}}'%self.P_0
        # # res_str += " &: 30 \\text{ minutes}\\ "
        # # res_str += " &: 40 \\frac{dollars}{hour}"
        # res_str += "\end{eqnarray*}"
        res_str = "\\begin{eqnarray*}"
        if title is not None:
            res_str += title + '\\\\'
        res_str += '\\underline{\\textrm{Societal Latency}}&'+'\\\\'
        res_str += '%.2f \\textrm{ } minutes&'%avg_l + '\\\\'
        res_str += '\\underline{\\textrm{Max Cost}}&'+ '\\\\' 
        res_str += '%d \\textrm{ }\\frac{dollars}{hour}&'%C_0
        res_str += "\end{eqnarray*}"


        ax_res.annotate(res_str, **anno_opts)
        plt.savefig(save_path)

    if sol_available:
        # Extact solution data first
        # flow_t, flow_d, demand_t
        ##---------------DRONES-----------------##
        # edge labels
        e_labels = {}
        first_label = True
        for edge in G_D.edges:
            if edge[0] == 0:
                if first_label:
                    e_labels[edge] = 'Flow: $%d$ $\\frac{drones}{hour}$'%flow_d[edge[1]-1]
                    first_label = False
                else:
                    e_labels[edge] = '$%d$'%flow_d[edge[1]-1]

        nodes = nx.draw_networkx_nodes(
            G = G_D, 
            pos = pos,
            ax = ax_drone,
            node_size = NODE_SIZE,
            node_color=NODE_COLOR,
        )
        edges = nx.draw_networkx_edges(
            G_D,
            pos,
            ax = ax_drone,
            node_size=NODE_SIZE,
            arrowstyle="->",
            arrowsize=ARROW_SIZE,
            width = EDGE_WIDTH,
            edge_color = EDGE_COLOR
        )
        labels = nx.draw_networkx_labels(
            G_D,
            pos,
            ax = ax_drone,
            labels=n_labels,
            font_size=NODE_FSIZE
        )
        edge_labels = nx.draw_networkx_edge_labels(
            G_D,
            pos,
            ax = ax_drone,
            font_size = EDGE_FSIZE,
            # label_pos = 0.5,
            edge_labels=e_labels,
            verticalalignment='center',
            horizontalalignment='center'
        )

        ##---------------TRUCKS-----------------##
        # edge labels, widths, and colors
        e_labels = {}
        e_widths = []
        e_colors = []
        first_label = True
        avg_p_l = 0
        avg_l = 0
        for edge in G.edges:
            e = edge_to_idx(edge)
            # labels
            if first_label:
                e_labels[edge] = 'Truck Flow: $%d$ $\\frac{cars}{hour}$'%flow_t[e]
                first_label = False
            else:
                e_labels[edge] = '$%d$'%flow_t[e]
            # widths and colors depend on num lanes

            e_widths.append(EDGE_WIDTH*lanes[e])
            new_l = l_road[e] + np.dot([omega[0][e],omega[1][e]],[flow_st[e],flow_c[e]+flow_t[e]]) 
            old_l = l_road[e] + np.dot([omega[0][e],omega[1][e]],[0,flow_c[e]]) 
            e_colors.append(new_l/old_l)
            avg_l += new_l*flow_c[e]
            avg_p_l += new_l*flow_st[e]

        avg_l /= Beta
        avg_p_l += np.dot(l_drone,flow_d)
        avg_p_l /= D_tot

        nodes = nx.draw_networkx_nodes(
            G = G, 
            pos = pos,
            ax = ax_truck,
            node_size = NODE_SIZE,
            node_color=NODE_COLOR,
        )
        edges = nx.draw_networkx_edges(
            G,
            pos,
            ax = ax_truck,
            node_size=NODE_SIZE,
            arrowstyle="->",
            arrowsize=ARROW_SIZE,
            width = e_widths,
            edge_color = e_colors,
            edge_cmap= cmap
        )
        labels = nx.draw_networkx_labels(
            G,
            pos,
            ax = ax_truck,
            labels=n_labels,
            font_size=NODE_FSIZE
        )
        edge_labels = nx.draw_networkx_edge_labels(
            G,
            pos,
            ax = ax_truck,
            font_size = EDGE_FSIZE,
            # label_pos = 0.5,
            edge_labels=e_labels,
            verticalalignment='center',
            horizontalalignment='center'
        )
        # colors
        pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        pc.set_array(e_colors)
        pc.set_clim(0,25)
        cb = plt.colorbar(pc,cax=ax_color)
        cb.outline.set_visible(False)
        cb.set_ticks([0,5,10,15,20,25])
        cb.set_ticklabels([0,5,10,15,20,25])
        ax_color.set_ylabel('Percent increase of latency')
        # RESULT BAR
        price = c_d*sum(D_v-(demand_t*p_per_truck)) + c_t*np.sum(demand_t) 

        anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                        va='center', ha='center')
        # create Latex like string for result
        res_str = "\\begin{eqnarray*}"
        if title is not None:
            res_str += title + '\\\\'
        res_str += '\\underline{\\textrm{Societal Latency}}&'+'\\\\' 
        res_str += '%.2f \\textrm{ } minutes&'%avg_l + '\\\\'
        res_str += '\\underline{\\textrm{Parcel Latency}}&'+'\\\\'
        res_str += '%.2f \\textrm{ } minutes&'%avg_p_l + '\\\\'
        res_str += '\\underline{\\textrm{Cost}}&' + '\\\\'
        res_str += '%d \\textrm{ }\\frac{dollars}{hour}&'%price
        res_str += "\end{eqnarray*}"


        ax_res.annotate(res_str, **anno_opts)
        plt.savefig(save_path)
    return None