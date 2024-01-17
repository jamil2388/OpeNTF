import datetime

import torch
import torch_geometric.utils
from torch import Tensor
import os
import sys
sys.path.append('..')
sys.path.append('../..')

from torch_geometric.data import download_url, extract_zip
import pickle
from torch.profiler import profile, ProfilerActivity
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric.transforms as T
import torch.nn.functional as F

import src.graph_params as graph_params
from src.mdl.graph_sage import Model as GSModel
from src.mdl.gcn import Model as GCNModel
from src.mdl.gat import Model as GATModel
from src.mdl.gin import Model as GINModel
from src.mdl.graph_sage_bk import Model as Model_bk
import tqdm as tqdm
import time
from sklearn.metrics import roc_auc_score
import logging
import argparse

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

    print(f'Loading data from filepath : {filepath}')
    logging.info(f'Loading data from filepath : {filepath}\n')

    # apply random initialization of features
    if(type(data) == HeteroData):
        for node_type in data.node_types:
            if(data[node_type].x.sum() == 0):
                print(f'Randomly assigning features to node type : {node_type}')
                data[node_type].x = torch.rand(data[node_type].x.shape)
    else:
        data.x = torch.rand(data.x.shape)

    # print(data)
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
        neg_sampling_ratio=graph_params.settings['model']['negative_sampling'],
        add_negative_train_samples=True,
        edge_types= edge_types,
        rev_edge_types=rev_edge_types,
    )

    train_data, val_data, test_data = transform(data)

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
        num_neighbors=[20, 10],
        neg_sampling_ratio=0.0,
        edge_label_index = edge_label_index_tuple,
        # edge_label_index = None,
        edge_label = edge_label,
        # edge_label = None,
        batch_size=32,
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

def create(data, model_name):

    if (model_name == 'gcn'):
        # gcn
        model = GCNModel(hidden_channels=hidden_channels, data=data)
    elif (model_name == 'gs'):
        # gs
        model = GSModel(hidden_channels=hidden_channels, data = data)
    elif (model_name == 'gat'):
        # gat
        model = GATModel(hidden_channels=hidden_channels, data = data)
    elif (model_name == 'gin'):
        # gin
        model = GINModel(hidden_channels=hidden_channels, data=data)

    print(model)
    print(f'\nDevice = {device}')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model,optimizer

# learn for unbatched data
def learn(data):
    start = time.time()
    is_directed = data.is_directed()
    min_loss = 100000000000
    epochs = graph_params.settings['model']['epochs']
    emb = {}

    # # adding profiler for experimental steps
    # ts = datetime.datetime.now()
    # profiler_log_dir = '../../data/preprocessed/logs'
    # if not os.path.isdir(profiler_log_dir): os.makedirs(profiler_log_dir)

    # prof = profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{profiler_log_dir}/gnn.{ts.day}.{ts.month}.{ts.year}_{ts.hour}.{ts.minute}.{ts.second}'),
    #     record_shapes=True,
    #     with_stack=True)

    # prof.start()

    for epoch in range(1, epochs + 1):
        # if(epoch > 1 + 1 + 3):
        #     break
        # prof.step()

        optimizer.zero_grad()
        data.to(device)
        pred = model(data, is_directed)

        if (type(data) == HeteroData):
            node_types = data.node_types
            edge_types = data.edge_types if is_directed else data.edge_types[
                                                                     :(len(data.edge_types)) // 2]
            # we have ground_truths per edge_label_index
            ground_truth = torch.empty(0).to(device)
            for edge_type in edge_types:
                ground_truth = torch.cat((ground_truth, data[edge_type].edge_label.unsqueeze(0)), dim=1)
            ground_truth = ground_truth.squeeze(0)

            for node_type in node_types:
                if(epoch == epochs):
                    emb[node_type] = model.x_dict[node_type]
            # ground_truth = sampled_data['user','rates','movie'].edge_label
        else:
            if (epoch == epochs):
                emb['node'] = model.x
            ground_truth = data.edge_label

        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()

        print(f'epoch : {epoch}, loss : {loss:.4f}, time elapsed : {(time.time() - start) / 60:.2f} min')

        if(loss < min_loss):
            min_loss = loss
        if(epoch % 10 == 0):
            print(f'epoch : {epoch}, loss : {loss:.4f}')
            logging.info(f'Epoch : {epoch}, Loss : {loss:.4f}')
            # auc = eval(val_data, 'validation')
    torch.save(model.state_dict(), f'{model_output}/{model_name}.ns{int(graph_params.settings["model"]["negative_sampling"])}.d{hidden_channels}.model.pt', pickle_protocol=4)
    if (type(data) == HeteroData):
        torch.save(model.x_dict, f'{model_output}/{model_name}.ns{int(graph_params.settings["model"]["negative_sampling"])}.d{hidden_channels}.emb.pt', pickle_protocol=4)
    else:
        torch.save(model.x, f'{model_output}/{model_name}.ns{int(graph_params.settings["model"]["negative_sampling"])}.d{hidden_channels}.emb.pt', pickle_protocol=4)

    print(f'\nEmbeddings saved to {model_output}/{model_name}.ns{int(graph_params.settings["model"]["negative_sampling"])}.d{hidden_channels}.emb.pt\n')
    logging.info(f'\nEmbeddings saved to {model_output}/{model_name}.ns{int(graph_params.settings["model"]["negative_sampling"])}.d{hidden_channels}.emb.pt\n')

    print(f'\nmin_loss after {epochs} epochs : {min_loss:.4f}\n')
    logging.info(f'\nMinimum loss after {epochs} epochs : {min_loss:.4f}\n')


    end = time.time()
    total_time = end - start
    print(f'total time taken : {total_time:.2f} seconds || {total_time / 60:.2f} minutes || {total_time / (60 * 60)} hours\n')
    logging.info(f'Total time : {total_time / 60:.2f} minutes || {total_time / (60 * 60)} hours\n')

    # prof.stop()

# learning with batching
def learn_batch(train_loader, is_directed):

    epochs = 1000

    for epoch in range(1, epochs + 1):
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


@torch.no_grad()
def eval(data, mode = 'validation'):
    model.eval()

    data.to(device)
    pred = model(data, is_directed)
    # The ground_truth and the pred shapes should be 1-dimensional
    # we squeeze them after generation
    if (type(data) == HeteroData):
        edge_types = data.edge_types if is_directed else data.edge_types[
                                                         :(len(data.edge_types)) // 2]
        # we have ground_truths per edge_label_index
        ground_truth = torch.empty(0).to(device)
        for edge_type in edge_types:
            ground_truth = torch.cat((ground_truth, data[edge_type].edge_label.unsqueeze(0)), dim=1)
        ground_truth = ground_truth.squeeze(0)
    else:
        ground_truth = data.edge_label

    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

    print(f'{mode} loss : {loss:.4f}')
    logging.info(f'{mode} loss : {loss:.4f}')

    print(f'pred = {pred}')
    print(f'ground_truth = {ground_truth}')
    print(f'pred = {pred.sigmoid()}')

    if(mode == 'validation'):
        auc = roc_auc_score(ground_truth, torch.sigmoid(pred))
        print()
        print(f"AUC : {auc:.4f}")
        return auc

# loader can be test or can be validation
def eval_batch(loader, is_directed):
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

def addargs(parser):
    parser.add_argument('-domains', nargs = '+', help = 'the domain of the dataset along with the version')
    parser.add_argument('-gnn_models', nargs = '+', help = 'the gnn models to use for embedding generation')
    parser.add_argument('-graph_types', nargs = '+', help = 'the graph types to use')
    parser.add_argument('-agg', nargs = '+', help = 'the aggregation types to use')
    parser.add_argument('-dim', type = int, help = 'the embedding dimension to use')
    parser.add_argument('-heads', type = int, help = 'the number of computational heads to use for gat models')
    parser.add_argument('--use_cpu', type = int, help = '1 if you want gat to use cpu')

    args = parser.parse_args()
    return args

def set_params(args):

    graph_params.settings['model']['hidden_channels'] = args.dim
    graph_params.settings['model']['gat']['heads'] = args.heads

# sample command including all the args
# cd OpeNTF_Jamil/src/mdl
# python gnn_j.py -domains dblp/dblp.v12.json.filtered.mt5.ts2 imdb/title.basics.tsv.filtered.mt5.ts2 -gnn_models gcn gs gin gat -graph_types m sm stm -agg none mean -dim 32 -heads 3

# same command in multiline
# (python gnn_j.py
# -domains dblp/dblp.v12.json.filtered.mt5.ts2 imdb/title.basics.tsv.filtered.mt5.ts2
# -gnn_models gcn gs gin gat
# -graph_types m sm stm
# -agg none mean
# -dim 32
# -heads 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN for OpeNTF')
    args = addargs(parser)

    set_params(args)
    hidden_channels = graph_params.settings['model']['hidden_channels']
    heads = graph_params.settings['model']['gat']['heads']

    # homogeneous_data = create_custom_homogeneous_data()
    # heterogeneous_data = create_custom_heterogeneous_data()

    # for domain in ['dblp/dblp.v12.json.filtered.mt5.ts2', 'imdb/title.basics.tsv.filtered.mt5.ts2']:
    for domain in args.domains:
    # for domain in ['dblp/dblp.v12.json.filtered.mt5.ts2']:
    # for domain in ['imdb/title.basics.tsv.filtered.mt5.ts2']:
    # for domain in ['uspt/patent.tsv.filtered.mt5.ts2']:
    # for domain in ['gith/data.csv.filtered.mt5.ts2']:
    # for domain in ['dblp/toy.dblp.v12.json']:

        log_filepath = f'../../data/preprocessed/{domain}'
        if not os.path.isdir(log_filepath): os.makedirs(log_filepath)

        logging.basicConfig(filename=f'{log_filepath}/emb.ns{graph_params.settings["model"]["negative_sampling"]}.h{heads}.d{hidden_channels}.log', format = '%(message)s', filemode = 'w', level=logging.INFO)
        logging.info(f'\n-------------------------------------')
        logging.info(f'-------------------------------------')
        logging.info(f'Domain/Data : {domain}')
        logging.info(f'-------------------------------------')
        logging.info(f'-------------------------------------\n')

        # for model_name in ['gcn', 'gs', 'gin', 'gat']:
        for model_name in args.gnn_models:
        # for model_name in ['gat']:
            for graph_type in args.graph_types:
            # for graph_type in ['m', 'sm', 'stm']:
            # for graph_type in ['sm']:
                for agg in args.agg:
                # for agg in ['none', 'mean']:
                # for agg in ['none']:
                # for agg in ['mean']:
                    if (model_name == 'gcn' and graph_type != 'm'):
                        continue

                    print(f'\nargs : {args}')

                    filepath = f'../../data/preprocessed/{domain}/gnn/{graph_type}.undir.{agg}.data.pkl'
                    model_output = f'../../data/preprocessed/{domain}/{model_name}/{graph_type}.undir.{agg}'
                    # model_output = f'../../data/preprocessed/{domain}/{model_name}'
                    if not os.path.isdir(model_output): os.makedirs(model_output)

                    logging.info(f'\n-------------------------------------------------------------------------------')
                    logging.info(f'Model : {model_name} || Graph Type : {graph_type} || Aggregation Type : {agg}')
                    logging.info(f'-------------------------------------------------------------------------------\n')

                    # load opentf datasets
                    # filepath = '../../data/preprocessed/dblp/toy.dblp.v12.json/gnn/stm.undir.mean.data.pkl'
                    # filepath = '../../data/preprocessed/dblp/dblp.v12.json.filtered.mt5.ts2/gnn/stm.undir.mean.data.pkl'
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

                    # set the device
                    if(args.use_cpu and model_name == 'gat' and graph_type in ['sm','stm'] and agg == 'none'):
                        device = torch.device('cpu')
                        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    else:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    print(f"Device: '{device}'")
                    torch.cuda.empty_cache()
                    train_data.to(device)

                    # the train_data is needed to collect info about the metadata
                    model,optimizer = create(train_data, model_name)

                    learn(train_data)
                    # the sampled_data from mini_batch_loader does not properly show the
                    # is_directed status
                    # learn_batch(train_loader, is_directed)
                    torch.cuda.empty_cache()
                    eval(test_data, 'test')
                    # eval_batch(test_loader, is_directed)
                    torch.cuda.empty_cache()
