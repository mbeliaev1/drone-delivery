from re import L
from tkinter import E
import matplotlib
import numpy as np
from numpy import linalg as la
import networkx as nx
import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt

def gnp_random_connected_digraph(n, p, seed):
    """
    Generates a random directed graph, 
    forcing it to be symmetric & complete, 
    then choosing additional edges w. prob p
    """
    edges = itertools.permutations(range(n), 2)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)

    for _, node_edges in itertools.groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        idx = seed.choice(len(node_edges))

        random_edge_in = node_edges[idx]
        G.add_edge(*random_edge_in) 
        random_edge_out = random_edge_in[::-1]
        G.add_edge(*random_edge_out)

        for e in node_edges:
            if seed.random() < p:
                G.add_edge(*e)
                G.add_edge(*e[::-1])
    return G
    
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def isPSD(B,epsilon):
    "Returns true is B is PSD to some tolerance epsilon"
    return sum(la.eigvals(B)>-epsilon)==B.shape[0]

def create_edge_maps(G):
    edges = list(G.edges)
    edge_to_idx = lambda edge: edges.index(edge)
    idx_to_edge = lambda idx: edges[idx]
    return edge_to_idx, idx_to_edge

def create_paths(G, cutoff):
    '''
    Input:
        G      - nx DiGraph
        cutoff - max length of paths to consider

    Outputs:
        paths  - tuple of paths 
    '''
    paths = ()
    for node in G.nodes:
        if node == 0:
            pass
        else:
            gen_paths = nx.shortest_simple_paths(G,0,node,'latency')
            for i in range(cutoff):
                try:
                    paths += (next(gen_paths),)
                except:
                    break
    return paths


def init_C_ops(G,m,n,paths,edge_to_idx):
    '''
    non convex version

    B: Creates matrix that transforms from paths to edges
       Output shapes: m x p 
    C: Creates matrix that transforms paths to their arrival nodes
       Output shapes: n-1 x p 
    D: Creates matrix that nodes to their downlink edges (outgoing)
       Output shapes: m x n-1 
    S: Creates matrix that converts from path flow to stopping trucks
       Output shapes: m x p
    
    '''
    p = len(paths)
    B = np.zeros((m,p),dtype=int)
    C = np.zeros((n-1,p),dtype=int)
    S = np.zeros((m,p),dtype=int)

    for i_path in range(p):
        # final edge in path
        end_node = paths[i_path][-1]
        C[end_node-1][i_path] = 1
        # set of edges in path
        for i in range(len(paths[i_path])-1):
            edge = tuple(paths[i_path][i:i+2])
            B[edge_to_idx(edge)][i_path] = 1
            S[edge_to_idx(edge)][i_path] = 1/len(paths[i_path])
    return B,C,S


def init_ops(G,m,n,paths,edge_to_idx):
    '''
    B: Creates matrix that transforms from paths to edges
       Output shapes: m x p 
    C: Creates matrix that transforms paths to their arrival nodes
       Output shapes: n-1 x p 
    D: Creates matrix that nodes to their downlink edges (outgoing)
       Output shapes: m x n-1 
    S: Creates matrix that converts from path flow to stopping trucks
       Output shapes: m x p
    
    '''
    p = len(paths)
    B = np.zeros((m,p),dtype=int)
    C = np.zeros((n-1,p),dtype=int)
    D = np.zeros((m,n-1),dtype=int)
    F = np.zeros((m,p),dtype=int)

    for i_path in range(p):
        # final edge in path
        end_edge = (paths[i_path][-2],paths[i_path][-1])
        F[edge_to_idx(end_edge)][i_path] = 1
        # final node of path
        end_node = paths[i_path][-1]
        C[end_node-1][i_path] = 1
        # set of edges in path
        for i in range(len(paths[i_path])-1):
            edge = tuple(paths[i_path][i:i+2])
            B[edge_to_idx(edge)][i_path] = 1

    for node in range(G.number_of_nodes()-1):
        # set of outgoing edges per node
        # count = 1 # start with 1 as we already have the initial edge
        for edge in G.out_edges(node+1):
            D[edge_to_idx(edge)][node] = 1
            # count += 1
        # D[:][node] /= count # scale by number of edges that contribute at this node
        # # dont forget to scale the final edge too 
        # F
    E = np.matmul(D,C)
    # dividing by sum of axes scales it
    S = (E+F)/np.sum(E+F,axis=0)
    return B,C,D,S

def convert_to_e(G,omega,flow_c,lanes,l_road,capacity):
    '''
    Converts dictionairy, flow matrix, and lane matrix
    to be compatible with outedge lists from G
    '''
    omega_0 = []
    omega_1 = []
    flow_ec = []
    lanes_e = []
    l_road_e = []
    caps = []
    for edge in G.edges:
        v = edge[0]
        w = edge[1]
        n_lanes = lanes[v][w]
        
        lanes_e.append(n_lanes)
        flow_ec.append(flow_c[v][w])
        l_road_e.append(l_road[v][w])

        omega_0.append(omega[n_lanes][0]*l_road[v][w]/capacity[v][w])
        omega_1.append(omega[n_lanes][1]*l_road[v][w]/capacity[v][w])
        caps.append(capacity[v][w])
    return np.asarray([omega_0,omega_1]), np.asarray(flow_ec), np.asarray(lanes_e), np.asarray(l_road_e), np.asarray(caps)

### FIGURE FUNCTIONS
def init_figure(fig_params, sol):
    '''
    sol is Bool that tells us if we should draw dual graph 
    or setup
    '''
    plt.rcParams.update(fig_params['rc_params'])
    # print(plt.rcParams["text.latex.preamble"])
    # print('something else')

        
    # create fig
    fig = plt.figure(figsize=(fig_params['setup']['figw'],fig_params['setup']['figh']))

    spec = fig.add_gridspec(ncols=len(fig_params['setup']['widths']), 
                            nrows=len(fig_params['setup']['heights']), 
                            width_ratios=fig_params['setup']['widths'],
                            height_ratios=fig_params['setup']['heights'],
                            left=fig_params['setup']['l'],
                            right=fig_params['setup']['r'],
                            top=fig_params['setup']['t'],
                            bottom=fig_params['setup']['b'])

    spec.update(wspace=fig_params['setup']['wspace'], 
                hspace=fig_params['setup']['hspace'])
    bbox = mpl.transforms.Bbox([[0,0],[fig_params['setup']['figw'],fig_params['setup']['figh']]])

    # SETUP #
    if not sol:
        # ax_drone = fig.add_subplot(spec[0,0])
        ax_truck = fig.add_subplot(spec[0,:])
        ax_color = fig.add_subplot(spec[1,1])

        # turn them off
        # ax_drone.axis('off')
        ax_truck.axis('off')
        # ax_res.axis('off')
        return(fig,ax_truck,ax_color,bbox)

    else:
        # ax_drone = fig.add_subplot(spec[0,0])
        ax_lsol = fig.add_subplot(spec[:,0])
        ax_rsol = fig.add_subplot(spec[:,3])
        ax_res = fig.add_subplot(spec[1,1:3])
        # ax_lres = fig.add_subplot(spec[2,1])
        # ax_rres = fig.add_subplot(spec[2,2])
        cax_edges = fig.add_subplot(spec[3,1:3])
        cax_nodes = fig.add_subplot(spec[2,1:3])
        # turn them off
        # ax_drone.axis('off')
        ax_lsol.axis('off')
        ax_rsol.axis('off')
        ax_res.axis('off')
        
        return(fig,ax_lsol,ax_rsol,ax_res,cax_edges,cax_nodes,bbox)


def draw_nodes(G,pos,ax,fig_params,n_labels,n_colors):

    # curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    # straight_edges = list(set(G.edges()) - set(curved_edges))
    nodes = nx.draw_networkx_nodes(
        G = G, 
        pos = pos,
        ax = ax,
        node_size = fig_params['N_SIZE'],
        node_color=n_colors,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax = ax,
        labels=n_labels,
        font_size=fig_params['N_FSIZE']
    )
    return nodes

def draw_truck_graph(G,pos,ax_truck,fig_params,e_widths,e_colors,n_labels,cmap):

    # curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    # straight_edges = list(set(G.edges()) - set(curved_edges))
    nx.draw_networkx_nodes(
        G = G, 
        pos = pos,
        ax = ax_truck,
        node_size = fig_params['N_SIZE'],
        node_color=fig_params['N_COLOR'],
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax = ax_truck,
        labels=n_labels,
        font_size=fig_params['N_FSIZE']
    )

    edges = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        ax = ax_truck,
        node_size=fig_params['N_SIZE'],
        arrowstyle="->",
        arrowsize=fig_params['ARROW_SIZE'],
        width = e_widths,
        edge_color = e_colors,
        edge_cmap= cmap,
        edge_vmin = fig_params['setup']['cmin'],
        edge_vmax = fig_params['setup']['cmax'],
        connectionstyle= 'arc3, rad ='+ str(fig_params['arc_rad'])
    )

    return edges

def draw_colorbar(ax,edges,cmap,fig_params):
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    # pc.set_array(e_colors)
    pc.set_clim(fig_params['setup']['cmin'],fig_params['setup']['cmax'])
    cb = plt.colorbar(pc,cax=ax,orientation='horizontal',ticks=fig_params['setup']['cticks'])
    cb.set_ticklabels(fig_params['setup']['cticklabels'])
    # cb.set_ticks(ticks=[0,100,200,300])
    # cb.outline.set_visible(False)
    # ax_color.xaxis.set_ticks_position("bottom")
    ax.set_xlabel(fig_params['setup']['clabel'])
    ax.xaxis.set_label_position("top")
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,
        pad=2,
        direction = 'in') # labels along the bottom edge are off

    return cb

def truncate_colormap(cmap, minval=0.0, maxval=1.0, N=101):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, 100)),N)
    # cmap_list = []
    # for i in range(new_cmap.N):
    #     rgba = cmap_list(i)
    #     # rgb2hex accepts rgb or rgba
    #     node_color_map.append(str(matplotlib.colors.rgb2hex(rgba)))
    return new_cmap


def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items