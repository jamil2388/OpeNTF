import torch
import torch_geometric.utils
from torch import Tensor
import os
from torch_geometric.data import download_url, extract_zip
import pandas as pd
import pickle
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric.transforms as T
import torch.nn.functional as F
from src.mdl.graph_sage import Model as GSModel
from src.mdl.gcn import Model as GCNModel
from src.mdl.graph_sage_bk import Model as Model_bk
import tqdm as tqdm
from sklearn.metrics import roc_auc_score

'''
Class definitions
'''



# draw any graph
def draw_graph(G):

    G = torch_geometric.utils.to_networkx(G.to_homogeneous())
    nx.draw(G, node_color = "red", node_size = 1000, with_labels = True)
    plt.margins(0.5)
    plt.show()

# load previously written graph files
def load_data(filepath):

    print(f'Load Data')
    print('------------------------------------------')
    print()


    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f'filepath : {filepath}')
    print(data)
    return data

def create_custom_homogeneous_data():
    data = Data()

    num_nodes = 8

    # create
    data.node_id = torch.arange(start=1, end=num_nodes + 1, step=1)
    print(f'data.num_nodes = {data.num_nodes}')

    # define edges u to m
    edge_index = torch.tensor([[1, 1], [1, 3], [1, 5], \
                                   [2, 3], [2, 5], [2, 6], \
                                   [3, 3], [3, 4], \
                                   [4, 1], [4, 2], [4, 5], [4, 7]], dtype=torch.long)
    # optional edge_weight
    edge_weight = torch.zeros((edge_index.shape[0],))
    edge_weight[3] = 1

    data.x = torch.tensor([[0]] * data.num_nodes, dtype = torch.float)
    data.edge_index = edge_index.t().contiguous()
    data.edge_weight = edge_weight
    print(f'data.num_edges = {data.num_edges}')
    print(f'data.edge_index = {data.edge_index}')

    # make the graph undirected, generate reverse edges
    # data = T.ToUndirected()(data)

    # review the created data
    print(f'data = {data}')

    data.validate(raise_on_error = True)
    return data

# create a custom heterogeneous data here
def create_custom_heterogeneous_data():

    data = HeteroData()

    num_user = 4
    num_movie = 7

    # create
    data["user"].node_id = torch.arange(start = 0, end = num_user, step = 1)
    data["movie"].node_id = torch.arange(start = 0, end = num_movie, step = 1)

    for node_type in data.node_types:
        data[node_type].x = torch.tensor([[0]] * data[node_type].num_nodes, dtype = torch.float)

    print(f'data["user"].num_nodes = {data["user"].num_nodes}')
    print(f'data["movie"].num_nodes = {data["movie"].num_nodes}')
    print(f'data["movie"].node_id = {data["user"].node_id}')
    print(f'data["movie"].node_id = {data["movie"].node_id}')

    # define edges
    # urm = user rates movie
    # urrm = user rev_rates movie
    # urm_edge_index = torch.tensor([[1, 1], [1, 3], [1, 5], \
    #                   [2, 3], [2, 5], [2, 6], \
    #                   [3, 3], [3, 4],  \
    #                   [4, 1], [4, 2], [4, 5], [4, 7]], dtype = torch.long)
    urm_edge_index = torch.tensor([[1, 1], [1, 3], [1, 5], \
                      [2, 3], [2, 5], [2, 6], \
                      [3, 3], [3, 4],  \
                      [4, 1], [4, 2], [4, 5], [4, 7]], dtype = torch.long)
    uhm_edge_index = torch.tensor([[1, 6], [1, 3], [1, 5], \
                      [2, 3], [2, 5], [2, 6], [2, 1],\
                      [3, 3], [3, 4],  \
                      [4, 1], [4, 3], [4, 5], [4, 7]], dtype = torch.long)
    urm_edge_index -= 1
    uhm_edge_index -= 1

    # optional weight matrix
    urm_edge_attr = torch.ones((urm_edge_index.shape[0], 1))
    uhm_edge_attr = torch.ones((uhm_edge_index.shape[0], 1))

    data["user", "rates", "movie"].edge_index = urm_edge_index.t().contiguous()
    data["user", "hates", "movie"].edge_index = uhm_edge_index.t().contiguous()
    data["user", "rates", "movie"].edge_attr = urm_edge_attr
    data["user", "hates", "movie"].edge_attr = uhm_edge_attr

    print(f'data["user", "rates", "movie"].num_edges = {data["user", "rates", "movie"].num_edges}')
    print(f'data["user", "rates", "movie"].edge_index = {data["user", "rates", "movie"].edge_index}')
    print(f'data["user", "rates", "movie"].urm_edge_attr = {data["user", "rates", "movie"].edge_attr}')

    # make the graph undirected
    data = T.ToUndirected()(data)

    # review the created data
    print(f'data = {data}')

    data.validate(raise_on_error = True)
    return data

# check the characteristics of the splits
def validate_splits(train_data, val_data, test_data):
    print()
    print(f'-------------------Checking the Splits-----------------------')
    print()
    print(f'The splits are : \n')
    print(f'Train Data : \n{train_data}')
    print(f'Val Data : \n{val_data}')
    print(f'Test Data : \n{test_data}')

    if(type(train_data) == Data):
        # convert to df to compare the overlaps
        train_df = pd.DataFrame(train_data.numpy())
        val_df = pd.DataFrame(val_data.numpy())
        test_df = pd.DataFrame(test_data.numpy())

        train_test_df = pd.merge(train_df, test_df, how = 'inner', on = train_data.columns)


# define the train, valid, test splits
def define_splits(data):

    if(type(data) == HeteroData):
        num_edge_types = len(data.edge_types)

        # directed graph means we dont have any reverse edges
        if(data.is_directed()):
            edge_types = data.edge_types
            rev_edge_types = None
        else :
            edge_types = data.edge_types[:num_edge_types // 2]
            rev_edge_types = data.edge_types[num_edge_types // 2:]
    else:
        edge_types = None
        rev_edge_types = None


    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types= edge_types,
        rev_edge_types=rev_edge_types,
    )

    # print(f'---------------Before Transform-------------')
    # print()
    # print(f'data["user"] = {data["user"].num_nodes}')
    # print(f'data["movie"] = {data["movie"].num_nodes}')
    # print(f'data["user","rates","movie"].edge_index = {data["user", "rates", "movie"].edge_index}')
    # print(f'data["user","rates","movie"].num_edges = {data["user", "rates", "movie"].num_edges}')
    # print(f'data["movie", "rev_rates", "user"].num_edges = {data["movie", "rev_rates", "user"].num_edges}')

    train_data, val_data, test_data = transform(data)

    # print()
    # print(f'---------------After Transform-------------')
    #
    # print("Training data:")
    # print("==============")
    # print(train_data)
    # print(f'........................................')
    # # the edges for supervision
    # print(f'train_data.edge_label_index and edge_index')
    # print(train_data["user", "rates", "movie"].edge_label_index)
    # print(train_data["user", "rates", "movie"].edge_index)
    # print(f'........................................')
    # print()
    # print("Validation data:")
    # print("================")
    # print(f'val_data.edge_label_index and edge_index')
    # print(val_data["user", "rates", "movie"].edge_label_index)
    # print(val_data["user", "rates", "movie"].edge_index)
    # print(f'........................................')
    # print()
    # print(f'test_data.edge_label_index and edge_index')
    # print(test_data["user", "rates", "movie"].edge_label_index)
    # print(test_data["user", "rates", "movie"].edge_index)
    #
    # print(f'-------------------------------------------')
    # print(f'counts')
    # print(f'train_data.num_edges = {train_data["user", "rates", "movie"].num_edges}')
    # print(f'train_data.reverse.num_edges = {train_data["movie", "rev_rates", "user"].num_edges}')
    # print(f'validation_data.num_edges = {val_data["user", "rates", "movie"].num_edges}')
    # print(f'test_data.num_edges = {test_data["user", "rates", "movie"].num_edges}')

    return train_data, val_data, test_data

def create_mini_batch_loader(data):
    # Define seed edges:
    # we pick only a single edge_type to feed edge_label_index (need to verify this approach)
    if (type(data) == HeteroData):
        edge_types = train_data.edge_types
        edge_label_index = train_data[edge_types[0]].edge_label_index
        edge_label = train_data[edge_types[0]].edge_label
        edge_label_index_tuple = (edge_types[0], edge_label_index)
    else:
        edge_label_index = train_data.edge_label_index
        edge_label = train_data.edge_label
        edge_label_index_tuple = edge_label_index
    print(f'edge_label stuffs : {edge_label_index}, {edge_label}')

    mini_batch_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[100,100,100],
        neg_sampling_ratio=0.0,
        edge_label_index = edge_label_index_tuple,
        # edge_label_index = None,
        edge_label = edge_label,
        # edge_label = None,
        batch_size=1000000000,
        # shuffle=True,
    )

    # mini_batch_loader = HGTLoader(
    #     data = data,
    #     # Sample 20 nodes per type
    #     num_samples = [20],
    #     # Use a batch size of 128 for sampling training nodes of type paper
    #     batch_size=128,
    #     input_nodes=edge_label_index_tuple,
    # )

    # Inspect a sample:
    # sampled_data = next(iter(train_loader))
    for i, data in enumerate(mini_batch_loader):
        print(f'sample data for iteration : {i}')
        print(data)
        print(f'---------------------------------------\n')
    return mini_batch_loader

def create(data):

    model = GSModel(hidden_channels=10, data = data)
    # model = GCNModel(hidden_channels=10, data = data)
    # model = Model_bk(hidden_channels=10, data = data)
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model,optimizer

# learn for unbatched data
def learn(data):

    is_directed = data.is_directed()
    min_loss = 100000000000
    epochs = 100

    for epoch in range(1, epochs):
        optimizer.zero_grad()
        data.to(device)
        pred = model(data, is_directed)

        if (type(data) == HeteroData):
            edge_types = data.edge_types if is_directed else data.edge_types[
                                                                     :(len(data.edge_types)) // 2]
            # we have ground_truths per edge_label_index
            ground_truth = torch.empty(0)
            for edge_type in edge_types:
                ground_truth = torch.cat((ground_truth, data[edge_type].edge_label.unsqueeze(0)), dim=1)
            ground_truth = ground_truth.squeeze(0)
            # ground_truth = sampled_data['user','rates','movie'].edge_label
        else:
            ground_truth = data.edge_label

        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()

        if(loss < min_loss):
            min_loss = loss
        if(epoch % 10 == 0):
            print(f'epoch : {epoch}, loss : {loss:.4f}')
    print(f'min_loss after {epochs} epochs : {min_loss:.4f}')

# learning with batching
def learn_batch(train_loader, is_directed):
    for epoch in range(1, 1000):
        total_loss = total_examples = 0
        # print(f'epoch = {epoch}')
        for sampled_data in train_loader:
            optimizer.zero_grad()

            sampled_data.to(device)
            pred = model(sampled_data, is_directed)

            # The ground_truth and the pred shapes should be 1-dimensional
            # we squeeze them after generation
            if(type(sampled_data) == HeteroData):
                edge_types = sampled_data.edge_types if is_directed else sampled_data.edge_types[:(len(sampled_data.edge_types)) // 2]
                # we have ground_truths per edge_label_index
                ground_truth = torch.empty(0)
                for edge_type in edge_types:
                    ground_truth = torch.cat((ground_truth, sampled_data[edge_type].edge_label.unsqueeze(0)), dim = 1)
                ground_truth = ground_truth.squeeze(0)
                # ground_truth = sampled_data['user','rates','movie'].edge_label
            else:
                ground_truth = sampled_data.edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            # print(f'loss = {loss}')
        # print(f'epoch = {epoch}')
        # print(f'loss = {loss}')
        # print(f'total_examples = {total_examples}')
        # print(f'total_loss = {total_loss}')

        # validation part here maybe ?
        if epoch % 10 == 0 :
            # auc = eval(val_loader)
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")


# loader can be test or can be validation
def eval(loader, is_directed):
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data, is_directed))
            # ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)
            if (type(sampled_data) == HeteroData):
                # we have ground_truths per edge_label_index
                ground_truths.append(sampled_data[edge_type].edge_label for edge_type in sampled_data.edge_types)
                # ground_truth = sampled_data['user','rates','movie'].edge_label
            else:
                ground_truths = sampled_data.edge_label

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"AUC: {auc:.4f}")
    return auc

if __name__ == '__main__':
    homogeneous_data = create_custom_homogeneous_data()
    heterogeneous_data = create_custom_heterogeneous_data()

    # load opentf datasets
    filepath = '../../data/preprocessed/dblp/toy.dblp.v12.json/gnn/stm.undir.none.data.pkl'
    data = load_data(filepath)
    is_directed = data.is_directed()

    # # draw the graph
    # draw_graph(data)

    # train_data, val_data, test_data = define_splits(homogeneous_data)
    # train_data, val_data, test_data = define_splits(heterogeneous_data)
    train_data, val_data, test_data = define_splits(data)
    # validate_splits(train_data, val_data, test_data)

    ## Sampling
    # train_loader = create_mini_batch_loader(train_data)
    # val_loader = create_mini_batch_loader(val_data)
    # test_loader = create_mini_batch_loader(test_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    # the train_data is needed to collect info about the metadata
    model,optimizer = create(train_data)
    # the sampled_data from mini_batch_loader does not properly show the
    # is_directed status
    learn(train_data)
    # learn_batch(train_loader, is_directed)
    # eval(test_loader)
