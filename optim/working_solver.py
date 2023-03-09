import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import gurobipy as gp
from helpers import *

class net_solver():
    def __init__(self, net_params,optim_params,fig_params,convex = False, integer=True):
        '''
        Inputs: 3 dicts of parameters, the main parameters are listed below
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
        self.convex = convex
        self.integer = integer
        # initialize networks
        self.G = nx.DiGraph(net_params['A'])
        self.m, self.n = self.G.number_of_edges(), self.G.number_of_nodes()
        self.G_D = nx.DiGraph(np.concatenate(
            ([0],np.ones(self.n-1),np.zeros(self.n*self.n-self.n))).reshape(self.n,self.n))
        self.edge_to_idx = lambda edge: list(self.G.edges).index(edge)

        self.omega, self.flow_c, self.lanes, self.l_road, self.capacity = convert_to_e(
                                        self.G,optim_params['omega'],
                                        net_params['flow_c'],
                                        net_params['lanes'],
                                        net_params['l_road'],
                                        net_params['capacity'])

        # create edge weights once you get latencies
        for edge in self.G.edges():
            nx.set_edge_attributes(self.G, {edge: {"latency": self.l_road[self.edge_to_idx(edge)]}})
        self.paths = create_paths(self.G, cutoff=optim_params['cutoff'])

        # initialize optim
        self.D_tot = np.sum(optim_params['D_v'])
        self.matB, self.matC, self.matD, self.matS = init_ops(
                                        self.G,
                                        self.m,
                                        self.n,
                                        self.paths,
                                        self.edge_to_idx)

        # convert data to edge format
        # init remainder of parameters required
        self.pos = net_params['pos']
        self.l_drone = net_params['l_drone']
        self.p_per_truck = optim_params['p_per_truck']
        self.Gamma = optim_params['Gamma']
        self.C_0 = optim_params['C_0']
        self.c_d = optim_params['c_d']
        self.c_t = optim_params['c_t']
        self.Beta = optim_params['Beta']
        self.D_v = optim_params['D_v']
        self.MIP_gap = optim_params['MIP_gap']
        # initialize the solver and other optim parameters 
        self.solver = self._init_gurobi()

        # save the figure params 
        self.fig_params = fig_params

    def _init_gurobi(self):
        '''
        Create gurobi optimizer (called at start)
        '''
        self.obj_Q = self.p_per_truck*self.Gamma/self.D_tot
        self.obj_a = self.p_per_truck*self.Gamma/self.D_tot
        self.obj_a *= self.matB.T@self.l_road - self.matC.T@self.l_drone
        # self.obj_a *= (self.matB.T@np.diag(self.omega[1])@self.flow_c - self.matC.T@self.l_drone)

        if self.convex:
            self.obj_Q *= self.matB.T@np.diag(self.omega[0])@self.matB + self.matB.T@np.diag(self.omega[1])@self.matB
            self.obj_a += ((1-self.Gamma)/self.Beta)*(self.flow_c@np.diag(self.omega[0])@self.matB + self.flow_c@np.diag(self.omega[1])@self.matB)
        else:
            self.obj_Q *= (self.matB.T@np.diag(self.omega[0])@self.matS + self.matB.T@np.diag(self.omega[1])@self.matB)
            self.obj_a += ((1-self.Gamma)/self.Beta)*(self.flow_c@np.diag(self.omega[0])@self.matS + self.flow_c@np.diag(self.omega[1])@self.matB)

        self.offset = (self.Gamma/self.D_tot)*(np.inner(self.l_drone,self.D_v)) + ((1-self.Gamma)/self.Beta)*np.inner(self.flow_c,self.l_road)
        # cost constraint params
        self.cost_cons = self.matC.T@np.ones(self.n-1)
        self.cost_ub = (self.C_0-self.c_d*self.D_tot)/(self.c_t-self.c_d*self.p_per_truck)
        # print(cost_ub)
        # demand constraint
        self.demand_cons = self.p_per_truck*self.matC
        self.demand_ub = self.D_v

        # create gurobi model
        solver = gp.Model()
        # intialize solver either GRB.INTEGER for MIQP or GRB.CONTINUOUS for QP 
        if self.integer:
            x = solver.addMVar(len(self.paths), lb=0, vtype=gp.GRB.INTEGER)
            solver.params.MIPGap = self.MIP_gap
        else:
            x = solver.addMVar(len(self.paths), lb=0, vtype=gp.GRB.CONTINUOUS)
            if not self.convex:
                solver.params.NonConvex = 2
                solver.params.MIPGap = self.MIP_gap
                
        solver.setObjective(x @ self.obj_Q @ x + self.obj_a@x, sense=gp.GRB.MINIMIZE)

        if self.c_t >= self.c_d*self.p_per_truck:
            solver.addConstr(self.cost_cons@x <= self.cost_ub)
        else:
            solver.addConstr(self.cost_cons@x >= self.cost_ub) 
        # solver.addConstr(self.cost_cons@x  self.cost_ub) 
        solver.addConstr(self.demand_cons@x <= self.demand_ub) 
        
        return solver

    def _convert_sol(self):
        '''
        Convert sol (used inside self.optimize())
        '''
        sol_dict = {}
        self.X = self.solver.X
        sol_dict['X'] = self.X

        # flow of trucks along edges
        sol_dict['flow_t'] = self.matB@self.X
        self.flow_t = sol_dict['flow_t']
        # flow of stopping trukcs along edges
        sol_dict['flow_st'] = self.matS@self.X
        self.flow_st = sol_dict['flow_st'] 
        # demand satisfied by trucks at ea node
        sol_dict['demand_t'] = (self.p_per_truck)*self.matC@self.X
        self.demand_t = sol_dict['demand_t']
        # flow of drones along their aerial paths to ea node
        sol_dict['flow_d'] = self.D_v - sol_dict['demand_t'] 
        self.flow_d = sol_dict['flow_d']
        # price constraint
        self.price = self.c_t*sum(self.X) + self.c_d*sum(self.flow_d)
        sol_dict['price'] = self.price
        
        # parcel and societal latency (as well as latency gain per edge)
        self.avg_p_l = 0
        self.avg_s_l = 0
        self.gain_l = []

        # get latency gain and save it for later
        edge_l = []
        for edge in self.G.edges:
            e = self.edge_to_idx(edge)
            new_l = self.l_road[e] + np.dot([self.omega[0][e],self.omega[1][e]],[self.flow_st[e],self.flow_c[e]+self.flow_t[e]]) 
            old_l = self.l_road[e] + np.dot([self.omega[0][e],self.omega[1][e]],[0,self.flow_c[e]]) 
            edge_l.append(new_l)
            self.gain_l.append(new_l/old_l)
            # can compute societal latency directly here
            self.avg_s_l += new_l*self.flow_c[e]

        # get parcel latency over path flows
        for i_path in range(len(self.paths)):
            p_l = 0
            for i in range(len(self.paths[i_path])-1):
                edge = tuple(self.paths[i_path][i:i+2])
                p_l += edge_l[self.edge_to_idx(edge)]
            self.avg_p_l += self.p_per_truck*p_l*self.X[i_path]

        self.avg_s_l /= self.Beta
        # add drone delivery
        self.avg_p_l += np.dot(self.l_drone,self.flow_d)
        self.avg_p_l /= self.D_tot

        sol_dict['avg_s_l'] = self.avg_s_l
        sol_dict['avg_p_l'] = self.avg_p_l

        
        return sol_dict

    def optimize(self):
        self.solver.optimize()
        self.solution = self._convert_sol()

    def visualize(self, title, sol_available, save_path):
        '''
        visualize and save the figure accordingly
        '''
        # setup the plotting environment
        cmap = plt.cm.Wistia
        fig, ax_truck, ax_drone, ax_color, ax_res = init_figure(self.fig_params)

        # NETWORK X #
        # for drone net, 0 central node connected to all others
        # Create node labels 
        n_labels = {}
        for i_node in range(len(self.pos)):
            n_labels[i_node] = '$v_{%d}$'%i_node

        if not sol_available:
            ##---------------DRONES-----------------##
            # edge labels
            e_labels = {}
            # first_label = True

            e_labels = {edge:'$%d$'%self.D_v[edge[1]-1] for edge in self.G_D.edges}
            # for edge in self.G_D.edges:
            #     if edge[0] == 0:
            #         if first_label:
            #             e_labels[edge] = 'Demand: $%d$ $\\frac{parcels}{hour}$'%self.D_v[edge[1]-1]
            #             first_label = False
            #         else:
            #             e_labels[edge] = '$%d$'%self.D_v[edge[1]-1]
            edges = draw_donre_graph(self.G_D,self.pos,ax_drone,self.fig_params,n_labels,e_labels)

            ##---------------TRUCKS-----------------##
            # edge labels, widths, and colors
            e_labels = {}
            e_widths = []
            e_colors = []
            first_label = True
            nom_l = 0

            for edge in self.G.edges:
                # labels
                e = self.edge_to_idx(edge)
                if first_label:
                    e_labels[edge] = 'Car Flow: $%d$ $\\frac{cars}{hour}$'%self.flow_c[e]
                    first_label = False
                else:
                    e_labels[edge] = '$%d$'%self.flow_c[e]
                # widths and colors depend on num lanes
                e_widths.append(self.fig_params['E_WIDTH']*self.lanes[e])
                l = self.l_road[e] + np.dot([self.omega[0][e],self.omega[1][e]],[0,self.flow_c[e]]) 
                # e_colors.append(l)
                e_colors.append(100*self.flow_c[e]/self.capacity[e])
                nom_l += l*self.flow_c[e]
            nom_l /= self.Beta

            e_colors = np.asarray(e_colors)
            edges = draw_edge_graph(self.G,self.pos,ax_truck,self.fig_params,e_widths,e_colors,e_labels,n_labels,cmap)

            # COLOR BAR
            pc = mpl.collections.PatchCollection(edges, cmap=cmap)
            pc.set_array(e_colors)
            pc.set_clim(0,200)
            cb = plt.colorbar(pc,cax=ax_color)
            cb.outline.set_visible(False)
            # cb.set_ticks([0,1,2,3,4])
            # cb.set_ticklabels([0,1,2,3,4])
            # ax_color.set_ylabel('$\\%$ of capacity $f_0$')
            ax_color.set_ylabel('Nominal Car Flow($\\%$ of $f^0$)')
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
            # if title is not None:
            res_str += title + '\\\\'
            res_str += '\\underline{\\textrm{Societal Latency}}&'+'\\\\'
            res_str += '%.2f \\textrm{ } minutes&'%nom_l + '\\\\'
            res_str += '\\underline{\\textrm{Max Cost}}&'+ '\\\\' 
            res_str += '%d \\textrm{ }\\frac{dollars}{hour}&'%self.C_0
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
            for edge in self.G_D.edges:
                if edge[0] == 0:
                    if first_label:
                        e_labels[edge] = 'Flow: $%d$ $\\frac{drones}{hour}$'%self.flow_d[edge[1]-1]
                        first_label = False
                    else:
                        e_labels[edge] = '$%d$'%self.flow_d[edge[1]-1]
            
            edges = draw_donre_graph(self.G_D,self.pos,ax_drone,self.fig_params,n_labels,e_labels)

            ##---------------TRUCKS-----------------##
            # edge labels, widths, and colors
            e_labels = {}
            e_widths = []
            e_colors = []
            first_label = True

            for edge in self.G.edges:
                e = self.edge_to_idx(edge)
                # labels
                if first_label:
                    e_labels[edge] = 'Truck Flow: $%d$ $\\frac{cars}{hour}$'%self.flow_t[e]
                    first_label = False
                else:
                    e_labels[edge] = '$%d$'%self.flow_t[e]
                e_widths.append(self.fig_params['E_WIDTH']*self.lanes[e])
                e_colors.append(self.gain_l[e])
            e_colors = np.asarray(e_colors)

            edges = draw_edge_graph(self.G,self.pos,ax_truck,self.fig_params,e_widths,e_colors,e_labels,n_labels,cmap)
            # colors
            pc = mpl.collections.PatchCollection(edges, cmap=cmap)
            pc.set_array(e_colors)
            pc.set_clim(0,25)
            cb = plt.colorbar(pc,cax=ax_color)
            cb.outline.set_visible(False)
            cb.set_ticks([0,5,10,15,20,25])
            cb.set_ticklabels([0,5,10,15,20,25])
            ax_color.set_ylabel('Percent increase of latency')

            anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                            va='center', ha='center')
            # create Latex like string for result
            res_str = "\\begin{eqnarray*}"
            if title is not None:
                res_str += title + '\\\\'
            res_str += '\\underline{\\textrm{Societal Latency}}&'+'\\\\' 
            res_str += '%.2f \\textrm{ } minutes&'%self.avg_s_l + '\\\\'
            res_str += '\\underline{\\textrm{Parcel Latency}}&'+'\\\\'
            res_str += '%.2f \\textrm{ } minutes&'%self.avg_p_l + '\\\\'
            res_str += '\\underline{\\textrm{Cost}}&' + '\\\\'
            res_str += '%d \\textrm{ }\\frac{dollars}{hour}&'%self.price
            res_str += "\end{eqnarray*}"
            ax_res.annotate(res_str, **anno_opts)
            plt.savefig(save_path)

        return None