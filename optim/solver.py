import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import gurobipy as gp
from optim.helpers import *

class net_solver():
    def __init__(self, net_params):
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
        # initialize networks
        self.G = nx.DiGraph(net_params['A'])
        self.m, self.n = self.G.number_of_edges(), self.G.number_of_nodes()
        self.G_D = nx.DiGraph(np.concatenate(
            ([0],np.ones(self.n-1),np.zeros(self.n*self.n-self.n))).reshape(self.n,self.n))
        self.edge_to_idx = lambda edge: list(self.G.edges).index(edge)

        self.omega, self.flow_c, self.lanes, self.l_road, self.capacity = convert_to_e(
                                        self.G,net_params['omega'],
                                        net_params['flow_c'],
                                        net_params['lanes'],    
                                        net_params['l_road'],
                                        net_params['capacity'])

        # create edge weights once you get latencies
        for edge in self.G.edges():
            e = self.edge_to_idx(edge)
            lat = self.l_road[e] + np.dot([self.omega[0][e],self.omega[1][e]],[0,self.flow_c[e]]) 
            nx.set_edge_attributes(self.G, {edge: {"latency": lat}})

        self.paths = create_paths(self.G, cutoff=net_params['cutoff'])

        # initialize operators
        self.matB, self.matC, self.matD, self.matS = init_ops(
                                        self.G,
                                        self.m,
                                        self.n,
                                        self.paths,
                                        self.edge_to_idx)

        # leftover parameters
        self.pos = net_params['pos']
        self.l_drone = net_params['l_drone']
        # self.l_nom = net_params['delivery_time']
        self.Beta = net_params['Beta']
        self.D_tot = np.sum(net_params['D_v'])
        self.D_v = net_params['D_v']
        
    def _init_gurobi(self):
        '''
        Create gurobi optimizer (called at start)
        matrices do not match with those defined in paper:
            matB = A (path to truck flow)
            matC = B (path to demand)
            matS = E (path to stopping truck)
        '''
        # main quadratic term 
        self.obj_Q = self.p_per_truck*self.Gamma/self.D_tot
        self.obj_Q *= (self.matB.T@np.diag(self.omega[0])@self.matS + self.matB.T@np.diag(self.omega[1])@self.matB)
        
        # main linear term
        self.obj_a = self.p_per_truck*self.Gamma/self.D_tot
        self.obj_a *= self.matB.T@self.l_road + self.matB.T@np.diag(self.omega[1])@self.flow_c- self.matC.T@self.l_drone
        self.obj_a += ((1-self.Gamma)/self.Beta)*(self.flow_c@np.diag(self.omega[0])@self.matS + self.flow_c@np.diag(self.omega[1])@self.matB)

        # intercept for objective fun
        self.offset = (self.Gamma/self.D_tot)*np.inner(self.l_drone,self.D_v)
        self.offset += ((1-self.Gamma)/self.Beta)*(np.inner(self.flow_c,self.l_road) + self.flow_c@np.diag(self.omega[1])@self.flow_c)

        # cost constraint params
        self.cost_cons = (self.c_t-self.c_d*self.p_per_truck)*self.matC.T@np.ones(self.n-1)
        self.cost_ub = self.C_0-self.c_d*self.D_tot

        # demand constraint
        self.demand_cons = self.p_per_truck*self.matC
        self.demand_ub = self.D_v

        # create gurobi model
        solver = gp.Model()

        # intialize solver as continuous QP (non convex with MIP gap preset)
        x = solver.addMVar(len(self.paths), lb=0, vtype=gp.GRB.CONTINUOUS)

        solver.setObjective(x @ self.obj_Q @ x + self.obj_a@x, sense=gp.GRB.MINIMIZE)
        solver.addConstr(self.cost_cons@x <= self.cost_ub)

        # change demand constraint to force trucks to satisfy demand if no drones
        if self.allow_drones:
            solver.addConstr(self.demand_cons@x <= self.demand_ub) 
        else:
            solver.addConstr(self.demand_cons@x == self.demand_ub) 

        # some extra inits for efficiency:
        solver.params.NonConvex = 2
        # solver.params.MIPGap = self.MIP_gap
        # solver.params.MIPFocus = 3
        # solver.params.NodefileStart = 0.5
        # solver.params.Threads = 8
        # if self.integer:
        #     x = solver.addMVar(len(self.paths), lb=0, vtype=gp.GRB.INTEGER)
        #     solver.params.MIPGap = self.MIP_gap
        # else:
        #     x = solver.addMVar(len(self.paths), lb=0, vtype=gp.GRB.CONTINUOUS)
        #     if not self.convex:
        #         solver.params.NonConvex = 2
        #         solver.params.MIPGap = self.MIP_gap
                


        # if self.c_t >= self.c_d*self.p_per_truck:
        #     solver.addConstr(self.cost_cons@x <= self.cost_ub)
        # else:
        #     solver.addConstr(self.cost_cons@x >= self.cost_ub) 
        # # solver.addConstr(self.cost_cons@x  self.cost_ub) 
        # solver.addConstr(self.demand_cons@x <= self.demand_ub) 

        return solver

    def _convert_sol(self):
        '''
        Convert sol (used inside self.optimize())
        '''
        sol_dict = {}
        # flow of trucks along paths X
        sol_dict['X'] = self.solver.X
        # flow of trucks along edges
        sol_dict['flow_t'] = self.matB@sol_dict['X']
        # flow of stopping trukcs along edges
        sol_dict['flow_st'] = self.matS@sol_dict['X']
        # demand satisfied by trucks at ea node
        sol_dict['demand_t'] = (self.p_per_truck)*self.matC@sol_dict['X']
        # flow of drones along their aerial paths to ea node
        sol_dict['flow_d'] = self.D_v - sol_dict['demand_t'] 
        # price constraint
        sol_dict['price'] = self.c_t*sum(sol_dict['X']) + self.c_d*sum(sol_dict['flow_d'])
        
        sol_dict['price_truck'] = self.c_t*sum(sol_dict['X'])
        sol_dict['price_drone'] = self.c_d*sum(sol_dict['flow_d'])  

        sol_dict['portion_truck'] = 100*((self.D_tot-np.sum(sol_dict['flow_d']))/self.D_tot)
        sol_dict['portion_drone'] = 100*((np.sum(sol_dict['flow_d']))/self.D_tot)

        # parcel and societal latency (as well as latency gain per edge)
        avg_p_l = 0
        avg_s_l = 0
        gain_l = []

        # get latency gain and save it for later
        edge_l = []
        for edge in self.G.edges:
            e = self.edge_to_idx(edge)
            new_l = self.l_road[e] + np.dot([self.omega[0][e],self.omega[1][e]],[sol_dict['flow_st'][e],self.flow_c[e]+sol_dict['flow_t'][e]]) 
            old_l = self.l_road[e] + np.dot([self.omega[0][e],self.omega[1][e]],[0,self.flow_c[e]]) 
            edge_l.append(new_l)
            gain_l.append(new_l/old_l)
            # gain_l.append(new_l/self.l_road[e])
            # can compute societal latency directly here
            avg_s_l += new_l*self.flow_c[e]
        avg_s_l /= self.Beta
        sol_dict['avg_s_l'] = avg_s_l
        sol_dict['gain_l'] = gain_l

        # get parcel latency over path flows
        for i_path in range(len(self.paths)):
            p_l = 0
            for i in range(len(self.paths[i_path])-1):
                edge = tuple(self.paths[i_path][i:i+2])
                p_l += edge_l[self.edge_to_idx(edge)]
            # p_l += self.l_nom
            avg_p_l += self.p_per_truck*p_l*sol_dict['X'][i_path]
        sol_dict['latency_truck'] = avg_p_l/(self.D_tot-np.sum(sol_dict['flow_d']))
        
        # add drone delivery
        if self.allow_drones:
            sol_dict['latency_drone'] = np.dot(self.l_drone,sol_dict['flow_d'])/np.sum(sol_dict['flow_d'])
            avg_p_l += np.dot(self.l_drone,sol_dict['flow_d'])
        else:
            sol_dict['latency_drone'] = 0
            
        avg_p_l /= self.D_tot
        sol_dict['avg_p_l'] = avg_p_l
        

        # additional
        sol_dict['runtime'] = self.solver.Runtime
        return sol_dict

    def optimize(self, optim_params):
        # convert data to edge format
        # init remainder of parameters required
        self.p_per_truck = optim_params['p_per_truck']
        self.Gamma = optim_params['Gamma']
        self.C_0 = optim_params['C_0']
        self.c_d = optim_params['c_d']
        self.c_t = optim_params['c_t']
        self.MIP_gap = optim_params['MIP_gap']
        self.allow_drones = optim_params['allow_drones']
        # initialize the solver and other optim parameters 
        self.solver = self._init_gurobi()
        self.solver.optimize()
        sol = self._convert_sol()

        return sol

    def visualize_setup(self, fig_params, save_path):
        '''
        visualize and save the figure accordingly
        '''
        # setup the plotting environment
        fig, ax_truck, ax_color, bbox = init_figure(fig_params, sol = False)
 
        # Create node labels 
        n_labels = {i_node:'$v_{%d}$'%i_node for i_node in range(len(self.pos))}

        # edge widths, and colors
        e_widths = [fig_params['E_WIDTH']*self.lanes[self.edge_to_idx(e)] for e in self.G.edges]
        e_colors = [100*self.flow_c[self.edge_to_idx(e)]/self.capacity[self.edge_to_idx(e)] for e in self.G.edges]
        e_colors = np.asarray(e_colors)

        # draw truck network and make colorbar
        temp = plt.cm.get_cmap('Oranges')    
        cmap = truncate_colormap(temp, fig_params['setup']['ctrunc'][0],fig_params['setup']['ctrunc'][1],N=101)

        edges = draw_truck_graph(self.G,self.pos,ax_truck,fig_params,e_widths,e_colors,n_labels,cmap)
        cb = draw_colorbar(ax_color,edges,cmap,fig_params)

        # set title and save
        fig.suptitle(fig_params['setup']['suptitle'])
        plt.savefig(save_path,bbox_inches=bbox)
        return None

    def visualize_sol(self, lsol_dict, rsol_dict, fig_params, save_path):
        '''
        visualize and save the dual sol figure
        '''
        # setup the plotting environment
        temp = plt.cm.get_cmap('Oranges')    
        cmap_edges = truncate_colormap(temp, 0.2,1,N=101)
        cmap_nodes = truncate_colormap(temp, fig_params['setup']['ctrunc'][0],fig_params['setup']['ctrunc'][1],N=101)
        # node_color_map = []
        # for i in range(cmap_nodes.N):
        #     rgba = cmap_nodes(i)
        #     # rgb2hex accepts rgb or rgba
        #     node_color_map.append(str(matplotlib.colors.rgb2hex(rgba)))

        fig,ax_lsol,ax_rsol,ax_res,cax_edges,cax_nodes,bbox = init_figure(fig_params, sol = True)

        # Create node labels 
        n_labels = {i_node:'$v_{%d}$'%i_node for i_node in range(len(self.pos))}

        all_edges = []
        all_colors = []
        # plot both graphs first and collect corresponding edges
        for ax_truck, sol_dict in zip([ax_lsol,ax_rsol],[lsol_dict,rsol_dict]):
            ##---------------TRUCKS-----------------##
            # edge labels, widths, and colors
            e_labels = {e:'$%d$'%sol_dict['flow_t'][self.edge_to_idx(e)] for e in self.G.edges}
            # e_labels[[0,1]] = 'Car Flow: $%d$ $\\frac{cars}{hour}$'%self.flow_c[self.edge_to_idx([0,1])]
            e_widths = [fig_params['E_WIDTH']*(self.lanes[self.edge_to_idx(e)]-1) for e in self.G.edges]
            e_colors = [100*(sol_dict['gain_l'][self.edge_to_idx(e)]) for e in self.G.edges]
            e_colors = np.asarray(e_colors)
            # edges = draw_truck_graph(self.G,self.pos,self.edge_to_idx,ax_truck,fig_params,e_widths,e_colors,e_labels,n_labels,cmap)
            edges = draw_truck_graph(self.G,self.pos,ax_truck,fig_params,e_widths,e_colors,n_labels,cmap_edges)
            all_colors.append(e_colors)
            all_edges.append(edges)

        ##---------------RESULT TAB-----------------##
        # # compute latency, and find the nominal societal latency
        # l = np.array(self.l_road,dtype=np.float64) 
        # l += np.array([np.dot([self.omega[0][e],self.omega[1][e]],[0,self.flow_c[e]]) for e in range(self.m)])
        # nom_l = np.inner(l,self.flow_c)/self.Beta
        # donre edge labels are demand for setup
        # e_labels = {e:'$%d$'%sol_dict['flow_d'][e[1]-1] for e in self.G_D.edges}
        # edges = draw_drone_graph(self.G_D,self.pos,ax_drone,fig_params,n_labels,e_labels)

        anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                        va='center', ha='center')
        # create Latex like string for result


        # lsol_dict['latency_truck']
        # lsol_dict['latency_drone']
        # lsol_dict['portion_truck']
        # lsol_dict['portion_drone']
        # lsol_dict['price_truck']
        # lsol_dict['price_drone']

        # rsol_dict['latency_truck']
        # rsol_dict['latency_drone']
        # rsol_dict['portion_truck']
        # rsol_dict['portion_drone']
        # rsol_dict['price_truck']
        # rsol_dict['price_drone']

        res_str  = r'''\centering'''
        res_str += r'''    \begin{tabular}{cc|c}'''
        res_str += r'''        \toprule'''
        res_str += r'''		$\textit{setup}$ & $\gamma=1$ & $\gamma=0$\\'''
        res_str += r'''  		\cmidrule(r){1-3}'''
        res_str += r'''  		& \multicolumn{2}{c}{latency (\textit{min})}\\'''
        res_str += r'''  		\cmidrule(r){1-3}'''
        res_str +=    '     \\textit{truck} &%.2f&%.2f'%(lsol_dict['latency_truck'],rsol_dict['latency_truck']) + '\\\\'
        res_str +=    '     \\textit{drone} &%.2f&%.2f'%(lsol_dict['latency_drone'],rsol_dict['latency_drone']) + '\\\\'
        res_str +=    '     \\textit{parcel} &%.2f&%.2f'%(lsol_dict['avg_p_l'],rsol_dict['avg_p_l']) + '\\\\'
        res_str +=    '     \\textit{societal} &%.2f&%.2f'%(lsol_dict['avg_s_l'],rsol_dict['avg_s_l']) + '\\\\'
        res_str += r'''		\cmidrule(r){1-3}'''
        res_str += r'''		& \multicolumn{2}{c}{price ($\frac{dollars}{hour}$)}\\'''
        res_str += r'''		\cmidrule(r){1-3}'''
        res_str +=    '     \\textit{truck} &%d&%d'%(round(lsol_dict['price_truck']),round(rsol_dict['price_truck'])) + '\\\\'
        res_str +=    '     \\textit{drone} &%d&%d'%(round(lsol_dict['price_drone']),round(rsol_dict['price_drone'])) + '\\\\'
        # res_str +=    '     \\textit{parcel} &%.2f&%.2f'%(lsol_dict['price'],rsol_dict['price']) + '\\\\'
        res_str += r'''		\cmidrule(r){1-3}'''
        res_str += r'''        \bottomrule'''
        res_str += r'''    \end{tabular}'''

        ax_res.annotate(res_str, **anno_opts)
        # all_colors = np.asarray(all_colors)
        print(min(all_colors[0]),min(all_colors[1]),max(all_colors[0]),max(all_colors[1]))
        
        ##---------------COLORS-----------------##
        # edges
        patches = []
        for i in range(2):
            for patch in all_edges[i]:
                patches.append(patch)

        cb_edges = draw_colorbar(cax_edges,patches,cmap_edges,fig_params)
        # nodes need colors
        l_n_colors = fig_params['N_COLOR'][:]
        r_n_colors = fig_params['N_COLOR'][:]

        for i_node in range(len(self.D_v)):
            # print(self.D_v[1])
            # print(l_n_colors[i_node+1])
            # print(lsol_dict['flow_d'][i_node])
            # print(rsol_dict['flow_d'][i_node])
            
            l_rgba = cmap_nodes(int(100*(1-lsol_dict['flow_d'][i_node]/self.D_v[1])))
            r_rgba = cmap_nodes(int(100*(1-rsol_dict['flow_d'][i_node]/self.D_v[1])))
            # rgb2hex accepts rgb or rgba
            # node_color_map.append()
            l_n_colors[i_node+1] = str(matplotlib.colors.rgb2hex(l_rgba))
            r_n_colors[i_node+1] = str(matplotlib.colors.rgb2hex(r_rgba))

        l_nodes = draw_nodes(self.G,self.pos,ax_lsol,fig_params,n_labels,l_n_colors)
        r_nodes = draw_nodes(self.G,self.pos,ax_rsol,fig_params,n_labels,r_n_colors)
        node_fig_dict = {}
        node_fig_dict['setup'] = {
        'cmin':0,
        'cmax':100,
        'cticks':[0,25,50,75,100],
        'cticklabels':['$0 \\%$','$25 \\%$','$50 \\%$','$75\\%$','$100\\%$'],
        'clabel':'Nodes: Truck Deliveries ($\\%$ of $\\mathcal{D}_v$)'}

        cb_nodes = draw_colorbar(cax_nodes,patches,cmap_nodes,node_fig_dict)
        # # print(min(colors),max(colors))
        # pc = mpl.collections.PatchCollection(patches, cmap=cmap)
        # pc.set_array(colors)
        # pc.set_clim(100,105)
        # cb = plt.colorbar(pc,cax=ax_color,orientation='horizontal',ticks=)
        # # cb.outline.set_visible(False)
        # # cb.set_ticks()
        # # cb.set_ticklabels(['$0 \\%$','$100 \\%$','$200 \\%$','$300\\%$'])

        # # ax_color.xaxis.set_ticks_position("bottom")
        # ax_color.set_xlabel()
        # ax_color.xaxis.set_label_position("top")
        # ax_color.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=True,      # ticks along the bottom edge are off
        #     top=False,
        #     pad=5,
        #     direction = 'in') # labels along the bottom edge are off
        fig.suptitle(fig_params['setup']['suptitle'])
        plt.savefig(save_path,bbox=bbox)


        return None


    # def visualize_sol_t(self, lsol_dict, rsol_dict, fig_params, save_path):
    #     '''
    #     visualize and save the dual sol figure
    #     '''
    #     # setup the plotting environment
    #     cmap_edges = plt.cm.Oranges
    #     cmap_nodes = plt.cm.Grays
        
    #     fig,ax_lsol,ax_rsol,ax_lres,ax_rres,ax_color,bbox = init_figure(fig_params, sol = True)

    #     # Create node labels 
    #     n_labels = {i_node:'$v_{%d}$'%i_node for i_node in range(len(self.pos))}

    #     all_edges = []
    #     all_colors = []
    #     # plot both graphs first and collect corresponding edges
    #     for ax_truck, ax_res, sol_dict in zip([ax_lsol,ax_rsol],[ax_lres,ax_rres],[lsol_dict,rsol_dict]):
    #         ##---------------TRUCKS-----------------##
    #         # edge labels, widths, and colors
    #         e_labels = {e:'$%d$'%sol_dict['flow_t'][self.edge_to_idx(e)] for e in self.G.edges}
    #         # e_labels[[0,1]] = 'Car Flow: $%d$ $\\frac{cars}{hour}$'%self.flow_c[self.edge_to_idx([0,1])]
    #         e_widths = [fig_params['E_WIDTH']*(self.lanes[self.edge_to_idx(e)]-1) for e in self.G.edges]
    #         e_colors = [100*(sol_dict['gain_l'][self.edge_to_idx(e)]) for e in self.G.edges]
    #         e_colors = np.asarray(e_colors)
    #         # edges = draw_truck_graph(self.G,self.pos,self.edge_to_idx,ax_truck,fig_params,e_widths,e_colors,e_labels,n_labels,cmap)
    #         edges = draw_truck_graph(self.G,self.pos,ax_truck,fig_params,e_widths,e_colors,n_labels,cmap)
    #         all_colors.append(e_colors)
    #         all_edges.append(edges)

    #         ##---------------RESULT TAB-----------------##
    #         # # compute latency, and find the nominal societal latency
    #         # l = np.array(self.l_road,dtype=np.float64) 
    #         # l += np.array([np.dot([self.omega[0][e],self.omega[1][e]],[0,self.flow_c[e]]) for e in range(self.m)])
    #         # nom_l = np.inner(l,self.flow_c)/self.Beta
    #         # donre edge labels are demand for setup
    #         e_labels = {e:'$%d$'%sol_dict['flow_d'][e[1]-1] for e in self.G_D.edges}
    #         # edges = draw_drone_graph(self.G_D,self.pos,ax_drone,fig_params,n_labels,e_labels)

    #         anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
    #                         va='center', ha='center')
    #         # create Latex like string for result
    #         res_str = "\\begin{eqnarray*}"
    #         # if title is not None:
    #         #     res_str += title + '\\\\'
    #         res_str += '\\underline{\\textrm{Societal Latency}}&'+'\\\\' 
    #         res_str += '%.2f \\textrm{ } minutes&'%sol_dict['avg_s_l'] + '\\\\'
    #         res_str += '\\underline{\\textrm{Parcel Latency}}&'+'\\\\'
    #         res_str += '%.2f \\textrm{ } minutes&'%sol_dict['avg_p_l'] + '\\\\'
    #         res_str += '\\underline{\\textrm{Cost}}&' + '\\\\'
    #         res_str += '%d \\textrm{ }\\frac{dollars}{hour}&'%sol_dict['price']
    #         res_str += "\end{eqnarray*}"
    #         ax_res.annotate(res_str, **anno_opts)
    #     all_colors = np.asarray(all_colors)
    #     print(min(all_colors[0]),min(all_colors[1]),max(all_colors[0]),max(all_colors[1]))
        
    #     ##---------------COLORS-----------------##
    #     patches = []
    #     # colors = []
    #     for i in range(2):
    #         for patch,color in zip(all_edges[i],all_colors[i]):
    #             patches.append(patch)
    #     #         colors.append(color)
    #     # colors = np.asarray(colors)

    #     cb = draw_colorbar(ax_color,patches,cmap,fig_params)
    #     # # print(min(colors),max(colors))
    #     # pc = mpl.collections.PatchCollection(patches, cmap=cmap)
    #     # pc.set_array(colors)
    #     # pc.set_clim(100,105)
    #     # cb = plt.colorbar(pc,cax=ax_color,orientation='horizontal',ticks=)
    #     # # cb.outline.set_visible(False)
    #     # # cb.set_ticks()
    #     # # cb.set_ticklabels(['$0 \\%$','$100 \\%$','$200 \\%$','$300\\%$'])

    #     # # ax_color.xaxis.set_ticks_position("bottom")
    #     # ax_color.set_xlabel()
    #     # ax_color.xaxis.set_label_position("top")
    #     # ax_color.tick_params(
    #     #     axis='x',          # changes apply to the x-axis
    #     #     which='both',      # both major and minor ticks are affected
    #     #     bottom=True,      # ticks along the bottom edge are off
    #     #     top=False,
    #     #     pad=5,
    #     #     direction = 'in') # labels along the bottom edge are off
    #     plt.savefig(save_path,bbox_inches=bbox)

    #     # compute latency, and find the nominal societal latency
    #     # l = np.array(self.l_road,dtype=np.float64) 
    #     # l += np.array([np.dot([self.omega[0][e],self.omega[1][e]],[0,self.flow_c[e]]) for e in range(self.m)])
    #     # nom_l = np.inner(l,self.flow_c)/self.Beta
    #     # donre edge labels are demand for setup
    #     # e_labels = {e:'$%d$'%self.D_v[e[1]-1] for e in self.G_D.edges}
    #     # edges = draw_drone_graph(self.G_D,self.pos,ax_drone,fig_params,n_labels,e_labels)

    #     # anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
    #     #                 va='center', ha='center')
    #     # create Latex like string for result
    #     # res_str = "\\begin{eqnarray*}"
    #     # if title is not None:
    #     # res_str += 'title' + '\\\\'
    #     # res_str += '\\underline{\\textrm{Societal Latency}}&'+'\\\\'
    #     # res_str += '%.2f \\textrm{ } minutes&'%nom_l + '\\\\'
    #     # res_str += '\\underline{\\textrm{Max Cost}}&'+ '\\\\' 
    #     # res_str += '%d \\textrm{ }\\frac{dollars}{hour}&'%self.C_0
    #     # res_str += "\end{eqnarray*}"


    #     # ax_res.annotate(res_str, **anno_opts)