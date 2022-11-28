import torch
import os
import math
from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.express as px
from tqdm import tqdm
import random
import pickle
import string
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
# from plotly import graph_objs as go
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import Data, download_url, extract_gz
from torch_geometric.nn import GAE, GCNConv, VGAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges, negative_sampling, degree
import plotly.express as px

# set the seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

data_path = "loc_df_75_3_skills.csv"

df = pd.read_csv(data_path)
print(df.head(), '\n')
df.info()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pkl():
    path_to_pickle = '../data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3'

    teams_pkl = path_to_pickle + '/teams.pkl'
    indexes_pkl = path_to_pickle + '/indexes.pkl'
    teamsvecs_pkl = path_to_pickle + '/teamsvecs.pkl'

    with open(teams_pkl, 'rb') as tb:
        teams = pickle.load(tb)

    with open(indexes_pkl, 'rb') as tb:
        indexes = pickle.load(tb)

    with open(teamsvecs_pkl, 'rb') as tb:
        teamsvecs = pickle.load(tb)

    return teams, indexes, teamsvecs

teams, indexes, teamsvecs = load_pkl()

def load_node_mapping(datafile_path, index_col, offset=0):
    """
    Maps each distinct node to a unique integer index.

    Args: datafile_path, string name of the tsv file containing the graph data
          index_col, string name of the column containing the nodes of concern
          offset, amount to shift the generated indexes by
    Returns: the mapping from node name to integer index
    """
    df = pd.read_csv(datafile_path, index_col=index_col)
    mapping = {index_id: i + offset for i, index_id in enumerate(df.index.unique())}
    return mapping


def load_edge_list(datafile_path, src_col, src_mapping, dst_col, dst_mapping):
    """
    Given node mappings, returns edge list in terms of node integer indices.

    Args: datafile_path, string name of the tsv file containing the graph data
          src_col, string name of the column corresponding to source nodes
          src_mapping, mapping from source node name to integer index
    Returns: the mapping from node name to integer index
    """
    df = pd.read_csv(datafile_path)
    src_nodes = [src_mapping[index] for index in df[src_col]]
    dst_nodes = [dst_mapping[index] for index in df[dst_col]]
    edge_index = torch.tensor([src_nodes, dst_nodes])
    return edge_index


def initialize_data(datafile_path, num_features=1):
    """
    Given a tsv file specifying disease-gene interactions, index the nodes and
    construct a Data object.
    """
    # Get disease node mapping and gene node mapping.
    # Each node type has its own set of integer ids.
    dz_col, gene_col = "loc_id", "skills"
    dz_mapping = load_node_mapping(datafile_path, dz_col, offset=0)
    gene_mapping = load_node_mapping(datafile_path, gene_col, offset=70)

    # Get edge index in terms of the integer indeces assigned to the nodes.
    edge_index = load_edge_list(
        datafile_path, dz_col, dz_mapping, gene_col, gene_mapping)

    # Add the reverse direction (aka make it a undirected graph)
    rev_edge_index = load_edge_list(
        datafile_path, gene_col, gene_mapping, dz_col, dz_mapping)

    # Construct a Data object.
    data = Data()
    data.num_nodes = len(dz_mapping) + len(gene_mapping)
    data.edge_index = torch.cat((edge_index, rev_edge_index), dim=1)
    # pretend we have uniform node features
    # data.x = torch.ones((data.num_nodes, num_features))

    return data, gene_mapping, dz_mapping


data_object, gene_mapping, dz_mapping = initialize_data(data_path)
print(data_object)
print("Number of skills:", len(gene_mapping))
print("Number of loc:", len(dz_mapping))


NUM_FEATURES =  25
data_object.x = torch.ones((data_object.num_nodes, NUM_FEATURES))

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

transform = T.Compose([T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True,
                      split_labels=True, add_negative_train_samples=True)])

# train_dataset = transform(data_object)
train_dataset, val_dataset, test_dataset = transform(data_object)
print("Train Data:\n", train_dataset)
# print("Validation Data:\n", val_dataset)
# print("Test Data:\n", test_dataset)

def get_edge_dot_products(data, model, num_dz_nodes=70):
  """
  A pair of nodes (u,v) is predicted to be connected with an edge if the dot
  product between the learned embeddings of u and v is high. This function
  computes and returns the dot product of all pairs of (dz_node, gene_node).

  Args:
    data, the data_object containing the original node featues
    model, the model that will be used to encode the data
    num_dz_nodes, the number of disease nodes; used to differentiate between
      disease and gene node embeddings
  Returns:
    dot_products, a numpy 2D array of shape (num_dz_nodes, num_gene_nodes)
      containing the dot product between each (dz_node, gene_node) pair.
  """
  model.eval()
  x = data.x
  z = model.encode(x, data.edge_index).detach().numpy()
  dz_z = z[:num_dz_nodes, :]
  gene_z = z[num_dz_nodes:, :]

  dot_products = np.einsum('ai,bi->ab', dz_z, gene_z)
  return dot_products   # numpy array of shape (num_dz_nodes, num_gene_nodes)


def get_ranked_edges(data_object, model, num_dz_nodes=70):
  """
  Ranks all potential edges as predicted by the model.

  Args:
    data, the data_object containing the original node featues
    model, the model that will be used to encode the data
    num_dz_nodes, the number of disease nodes; used to differentiate between
      disease and gene node embeddings
  Returns:
    ranked_edge_list, a full edge list ranked by the likelihood of the edge
      being a positive edge, in decreasing order
    ranked_dot_products, a list of the dot products of each edge's node
      embeddings, ranked in decreasing order
  """
  # Get dot products
  edge_dot_products = get_edge_dot_products(data_object, model, num_dz_nodes=num_dz_nodes)
  num_possible_edges = edge_dot_products.shape[0] * edge_dot_products.shape[1]

  # Get indeces, ranked by dot product in descending order. This is a tuple (indeces[0], indeces[1]).
  ranked_edges = np.unravel_index(np.argsort(-1 * edge_dot_products, axis=None), edge_dot_products.shape)
  assert len(ranked_edges[0]) == num_possible_edges

  # Get the corresponding, ranked edge list and ranked dot products. Note that
  # we need to add an offset for the gene_node indeces.
  offset = np.array([np.zeros(num_possible_edges, dtype=int), num_dz_nodes + np.ones(num_possible_edges, dtype=int)]).T
  ranked_edge_list = np.dstack(ranked_edges)[0] + offset
  assert ranked_edge_list.shape[0] == num_possible_edges

  # Get the corresponding ranked dot products
  ranked_dot_products = edge_dot_products[ranked_edges]
  assert ranked_dot_products.shape[0] == num_possible_edges

  return ranked_edge_list, ranked_dot_products

HIDDEN_SIZE = 200  #@param {type: "integer"}

OUT_CHANNELS = 69750  #@param {type: "integer"}

EPOCHS = 20#@param {type: "integer"}


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = GAE(GNN(NUM_FEATURES, HIDDEN_SIZE, OUT_CHANNELS))
model = model.to(device)

print(model)


def gae_train(train_data, model, optimizer):
    model.train()
    # gae_model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    # z = gae_model.encode(train_data.x, train_data.edge_index)
    print('#############################')
    print(type(z))
    print('Shape of Z,', z.shape)
    print('Value of Z', z)
    print('#############################')
    # loss = gae_model.recon_loss(z, train_data.pos_edge_label_index.to(device))
    loss = model.recon_loss(z, train_data.pos_edge_label_index.to(device))
    loss.backward(retain_graph=True)
    optimizer.step()
    return z, float(loss)

losses = []
test_auc = []
test_ap = []
train_aucs = []
train_aps = []

avg_test_auc = []
avg_test_ap = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

z = torch.empty((160908, 100))

for epoch in range(1, EPOCHS + 1):
    z, loss = gae_train(train_dataset, model, optimizer)
    losses.append(loss)

    # train_auc, train_ap = gae_test(train_dataset, model)

    # train_aucs.append(train_auc)
    # train_aps.append(train_ap)
    print('Epoch: {:03d}, loss:{:.4f}'.format(
        epoch, loss))
    # print('Epoch: {:03d}, test AUC: {:.4f}, test AP: {:.4f}, train AUC: {:.4f}, train AP: {:.4f}, loss:{:.4f}'.format(epoch, auc, ap, train_auc, train_ap, loss))

print('Type of Z', type(z))
print('Shape of Z', z.shape)
print('Value of Z', z)

print(z[0:70,])

loc_file = torch.load('loc_gnn_embeddings_v2.pkl')
skills_file = torch.load('skills_gnn_embeddings_v2.pkl')

one_big_matrix = np.zeros((165496, 100))
# locs_dict = {}
rows_skills = skills_file.shape[0]
rows_locs = loc_file.shape[0]
print(skills_file.shape)

for ix_k, t in tqdm(enumerate(teams.values()), total=len(teams.values())):
    one_small_vector = np.zeros((1, 100))
    # one_skills_vector = np.zeros((69679, 100))
    try:
        for s in t.skills:
            ix_s = indexes['s2i'][s]
            # print(skills_file[ix_s].shape)
            one_small_vector += skills_file[ix_s].cpu().detach().numpy().reshape((1,100))
            # one_small_vector += skills_file[ix_s].cpu().detach().numpy().reshape((1, 100))

        for l in t.members_details:
            loc = l[2]
            ix_l = indexes['l2i'][loc]
            one_small_vector += loc_file[ix_l].cpu().detach().numpy().reshape((1,100))

    except:
        pass
    one_big_matrix[ix_k] = one_small_vector

with open(f'one_big_matrix.pkl', "wb") as outfile: pickle.dump(one_big_matrix, outfile)
print(one_big_matrix.shape)
# torch.save(z[0:71,], 'loc_gnn_embeddings_v2.pkl')
# torch.save(z[71:,], 'skills_gnn_embeddings_v2.pkl')

# def initialize_data(datafile_path, num_features=1):
#     """
#     Given a tsv file specifying disease-gene interactions, index the nodes and
#     construct a Data object.
#     """
#     # Get disease node mapping and gene node mapping.
#     # Each node type has its own set of integer ids.
#     dz_col, gene_col = "loc_id", "ex_id"
#     dz_mapping = load_node_mapping(datafile_path, dz_col, offset=0)
#     gene_mapping = load_node_mapping(datafile_path, gene_col, offset=66)
#
#     # Get edge index in terms of the integer indeces assigned to the nodes.
#     edge_index = load_edge_list(
#         datafile_path, dz_col, dz_mapping, gene_col, gene_mapping)
#
#     # Add the reverse direction (aka make it a undirected graph)
#     rev_edge_index = load_edge_list(
#         datafile_path, gene_col, gene_mapping, dz_col, dz_mapping)
#
#     # Construct a Data object.
#     df = pd.read_csv(datafile_path)
#     data_h = HeteroData()
#     # data_h['loc_id'].x = df['loc_id']
#     # data_h['ex_id'].x = df['ex_id']
#     # print(data_h)
#     # data = Data()
#     data_h.num_nodes = len(dz_mapping) + len(gene_mapping)
#     data_h.edge_index = torch.cat((edge_index, rev_edge_index), dim=1)
#     # pretend we have uniform node features
#     # data_h.x = torch.ones((len(dz_mapping) + len(gene_mapping), num_features))
#
#     return data_h, gene_mapping, dz_mapping
#
#
# data_object, gene_mapping, dz_mapping = initialize_data(data_path)
# print(data_object)
# print("Number of expert:", len(gene_mapping))
# print("Number of loc:", len(dz_mapping))
# print(data_object.node_types)
# print(data_object.edge_types)
