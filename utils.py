from collections import defaultdict
from itertools import combinations
import pandas as pd
import torch
import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set as mis
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv
from itertools import chain, islice
from time import time


# GNN class to be instantiated with specified param values
class GCN_2L_Model(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, device):
        """
        Initialize a new instance of the core GCN model of provided size.
        Dropout is added in forward step.

        Inputs:
            in_feats: Dimension of the input (embedding) layer
            hidden_size: Hidden layer size
            dropout: Fraction of dropout to add between intermediate layer. Value is cached for later use.
            device: Specifies device (CPU vs GPU) to load variables onto
        """
        super().__init__()

        self.dropout_frac = dropout
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, number_classes).to(device)

    def forward(self, g, inputs):
        """
        Run forward propagation step of instantiated model.

        Input:
            self: GCN_dev instance
            g: DGL graph object, i.e. problem definition
            inputs: Input (embedding) layer weights, to be propagated through network
        Output:
            h: Output layer weights
        """

        # input step
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(g, h)
        h = torch.sigmoid(h)

        return h
    

class SAGE_2L_Model(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, device):
        super().__init__()

        self.dropout_frac = dropout
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='pool').to(device)
        self.conv2 = SAGEConv(hidden_size, number_classes, aggregator_type='pool').to(device)

    def forward(self, g, inputs):
        # input step
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(g, h)
        h = torch.sigmoid(h)

        return h
    

class GAT_2L_Model(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, device, num_heads):
        super().__init__()

        self.dropout_frac = dropout
        self.num_heads = num_heads
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=self.num_heads).to(device)
        self.conv2 = GATConv(hidden_size*self.num_heads, number_classes, num_heads=self.num_heads).to(device)

    def forward(self, g, inputs):
        # input step
        h = self.conv1(g, inputs)
        h = h.reshape(h.size()[0], -1) # reshape
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(g, h)
        h = h.reshape(h.size()[0], -1) # reshape
        h = torch.sigmoid(h)

        return h


# GNN class to be instantiated with specified param values
class GCN_1L_Model(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, device):
        super().__init__()

        self.dropout_frac = dropout
        # self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.decoder1 = nn.Linear(hidden_size, number_classes).to(device)


    def forward(self, g, inputs):
        # input step
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.decoder1(h)
        h = torch.sigmoid(h)

        return h
    

# GNN class to be instantiated with specified param values
class SAGE_1L_Model(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, device):
        super().__init__()

        self.dropout_frac = dropout
        # self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='pool').to(device)
        self.decoder1 = nn.Linear(hidden_size, number_classes).to(device)


    def forward(self, g, inputs):

        # input step
        h = self.conv1(g, inputs)
        print(f"Conv1: {h.size()}")
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.decoder1(h)
        h = torch.sigmoid(h)

        return h
    
# GNN class to be instantiated with specified param values
class GAT_1L_Model(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, device, num_heads):
        super().__init__()

        self.dropout_frac = dropout
        self.num_heads = num_heads
        # self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=self.num_heads).to(device)
        self.decoder1 = nn.Linear(hidden_size*self.num_heads, number_classes).to(device)

    def forward(self, g, inputs):

        # input step
        h = self.conv1(g, inputs)
        h = h.reshape(h.size()[0], -1) # reshape
        # print(f"Conv1: {h.size()}")
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.decoder1(h)
        h = torch.sigmoid(h)

        return h


# Generate random graph of specified size and type,
# with specified degree (d) or edge probability (p)
def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    
    # Need to pull nx graph into OrderedGraph so training will work properly
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    return nx_graph


# helper function to convert Q dictionary to torch tensor
def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.

    Input:
        Q: QUBO matrix as defaultdict
        nx_G: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        Q: QUBO as torch tensor
    """

    # get number of nodes
    n_nodes = len(nx_G.nodes)

    # get QUBO Q as torch tensor
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat


# Chunk long list
def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


# helper function for custom loss according to Q matrix
def loss_func(probs, Q_mat):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    probs_ = torch.unsqueeze(probs, 1)

    # minimize cost = x.T * Q * x
    cost = (probs_.T @ Q_mat @ probs_).squeeze()

    return cost


# # Construct graph to learn on
def get_gnn(n_nodes, params, model_type, torch_device, torch_dtype):
    """
    Generate GNN instance with specified structure. Creates GNN, retrieves embedding layer,
    and instantiates ADAM optimizer given those.

    Input:
        n_nodes: Problem size (number of nodes in graph)
        gnn_hypers: Hyperparameters relevant to GNN structure
        opt_params: Hyperparameters relevant to ADAM optimizer
        torch_device: Whether to load pytorch variables onto CPU or GPU
        torch_dtype: Datatype to use for pytorch variables
    Output:
        net: GNN instance
        embed: Embedding layer to use as input to GNN
        optimizer: ADAM optimizer instance
    """
    dim_embedding = params['dim_embedding']
    hidden_dim = params['hidden_dim']
    dropout = params['dropout']
    number_classes = params['number_classes']
    opt_params = {'lr': params['lr']}

    # instantiate the GNN
    if model_type == "GCN_2L_Model":
        net = GCN_2L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    
    elif model_type == "SAGE_2L_Model":
        net = SAGE_2L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    
    elif model_type == "GAT_2L_1H_Model":
        net = GAT_2L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device, 1)

    elif model_type == "GAT_2L_2H_Model":
        net = GAT_2L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device, 2)
    
    elif model_type == "GAT_2L_4H_Model":
        net = GAT_2L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device, 4)
    
    elif model_type == "GCN_1L_Model":
        net = GCN_1L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    
    elif model_type == "SAGE_1L_Model":
        net = SAGE_1L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device)

    elif model_type == "GAT_1L_1H_Model":
        net = GAT_1L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device, 1)

    elif model_type == "GAT_1L_2H_Model":
        net = GAT_1L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device, 2)    

    elif model_type == "GAT_1L_4H_Model":
        net = GAT_1L_Model(dim_embedding, hidden_dim, number_classes, dropout, torch_device, 4)        
    
    else:
        raise KeyError(f"{model_type} is an invalid model_type!")
    
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    # optimizer = torch.optim.Adam(lr=lr)

    return net, embed, optimizer


# Parent function to run GNN training given input config
def run_gnn_training(q_torch, dgl_graph, net, embed, optimizer, number_epochs, tol, patience, prob_threshold):
    """
    Wrapper function to run and monitor GNN training. Includes early stopping.
    """
    # Assign variable for user reference
    inputs = embed.weight

    prev_loss = 1.  # initial loss value (arbitrary)
    count = 0       # track number times early stopping is triggered

    # initialize optimal solution
    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),)).type(q_torch.dtype).to(q_torch.device)
    best_loss = loss_func(best_bitstring.float(), q_torch)

    # training history
    loss_hist = []
    epoch_hist = []

    t_gnn_start = time()

    # Training logic
    for epoch in range(number_epochs):

        # get logits/activations
        probs = net(dgl_graph, inputs)[:, 0]  # collapse extra dimension output from model

        # build cost value with QUBO cost function
        loss = loss_func(probs, q_torch)
        loss_ = loss.detach().item()

        # Apply projection
        bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')
            loss_hist.append(loss_)
            epoch_hist.append(epoch)

        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break

        # update loss tracking
        prev_loss = loss_

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()        # calculate gradient through compute graph
        optimizer.step()       # take step, update weights

    t_gnn = time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_}')
    print(f'GNN best continuous loss: {best_loss}')

    final_bitstring = (probs.detach() >= prob_threshold) * 1

    return net, epoch, final_bitstring, best_bitstring, best_loss, inputs, loss_hist, epoch_hist


def build_nx_graph(df_corr:pd.DataFrame, threshold:float) -> nx.Graph:
    """ Returns a Networkx.Graph from a correlation matrix.
        Args:
            df_corr (pd.DataFrame): Correlation matrix
            threshold (float): edges with an correlation coefficient smaller
                than the treshold will be removed. 
                Note: The treshold bust be choosen reasonalbe, the graph must
                not be disconnected.
        Returns:
            nx.Graph
    """
    cor_matrix = df_corr.values.astype('float')
    sim_matrix = 1 - cor_matrix
    G = nx.from_numpy_array(sim_matrix)
    H = G.copy()

    for (u, v, wt) in G.edges.data('weight'):

        if wt >= 1 - threshold:
            H.remove_edge(u, v)

        if u == v:
            H.remove_edge(u, v)

    return H


# helper function to generate Q matrix for Maximum Independent Set problem (MIS)
def gen_q_dict_mis(nx_G, penalty=2):
    """
    Helper function to generate QUBO matrix for MIS as minimization problem.
    
    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_dic = defaultdict(int)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty

    # all diagonal terms get -1
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -1

    return Q_dic


def gen_q_dict_mis_sharpe_reward(nx_G, rewards:list, penalty=2):
    """
    Helper function to generate QUBO matrix for MIS as minimization problem.
    
    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_dic = defaultdict(int)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty

    # all diagonal terms get -1
    for u, r in zip(nx_G.nodes, rewards):
        Q_dic[(u, u)] = r

    return Q_dic


# Run classical MIS solver (provided by NetworkX)
def run_mis_solver(nx_graph):
    """
    helper function to run traditional solver for MIS.
    
    Input:
        nx_graph: networkx Graph object
    Output:
        ind_set_bitstring_nx: bitstring solution as list
        ind_set_nx_size: size of independent set (int)
        number_violations: number of violations of ind.set condition
    """
    # compare with traditional solver
    t_start = time()
    ind_set_nx = mis(nx_graph)
    t_solve = time() - t_start
    ind_set_nx_size = len(ind_set_nx)

    # get bitstring list
    nx_bitstring = [1 if (node in ind_set_nx) else 0 for node in sorted(list(nx_graph.nodes))]
    edge_set = set(list(nx_graph.edges))

    # Updated to be able to handle larger scale
    print('Calculating violations...')
    # check for violations
    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set_nx, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))

    return nx_bitstring, ind_set_nx_size, number_violations, t_solve


# Calculate results given bitstring and graph definition, includes check for violations
def postprocess_gnn_mis(best_bitstring, nx_graph):
    """
    helper function to postprocess MIS results

    Input:
        best_bitstring: bitstring as torch tensor
    Output:
        size_mis: Size of MIS (int)
        ind_set: MIS (list of integers)
        number_violations: number of violations of ind.set condition
    """

    # get bitstring as list
    bitstring_list = list(best_bitstring)

    # compute cost
    size_mis = sum(bitstring_list)

    # get independent set
    ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    edge_set = set(list(nx_graph.edges))

    print('Calculating violations...')
    # check for violations
    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))

    return size_mis, ind_set, number_violations
