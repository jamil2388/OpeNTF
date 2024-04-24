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

# load previously written graph files
def load_data(filepath):

    print(f'Load Data')
    print('------------------------------------------')
    print()


    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f'Loading data from filepath : {filepath}')
    logging.info(f'Loading data from filepath : {filepath}\n')

    # print(data)
    return data

def plot_graph(x, y, xlabel, ylabel, title, fig_output):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    print(f'\nsaving figure as : {fig_output}\n')
    plt.savefig(fig_output)
    plt.clf()
    # plot.show()

def check_split(split_data, main_data, edge_type, index_type):
    d = {}
    for id1, (i, j) in enumerate(split_data[edge_type].edge_index.t() if index_type == 'edge_index' else split_data[edge_type].edge_label_index.t()):
        if(index_type == 'edge_label_index' and split_data[edge_type].edge_label[id1] == 0):
            continue
        # print(f'{id1}, {i}, {j}')
        for id2, (k, l) in enumerate(main_data[edge_type].edge_index.t()):
            if (i == k and j == l):
                d[i, j] = 1
            if (id2 == main_data[edge_type].edge_index.shape[1] - 1):
                if (d.get((i, j)) is None):
                    print(f'{i}, {j} from {index_type} not found in main data')
                    break
    print(f'd item count : {len(d.keys())}')

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
        num_test=0.0,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=ns,
        add_negative_train_samples=False,
        edge_types= edge_types,
        rev_edge_types=rev_edge_types,
    )

    train_data, val_data, test_data = transform(data)

    train_data.validate(raise_on_error = True)
    val_data.validate(raise_on_error = True)
    test_data.validate(raise_on_error = True)

    # check split
    # print(f'validating splits')
    # for edge_type in data.edge_types[:(len(data.edge_types) // 2)]:
    #     for split_data in [train_data, val_data, test_data]:
    #         for index_type in ['edge_index', 'edge_label_index']:
    #             check_split(split_data, data, edge_type, index_type)

    train_data.validate(raise_on_error = True)
    val_data.validate(raise_on_error = True)
    test_data.validate(raise_on_error = True)

    return train_data, val_data, test_data, edge_types, rev_edge_types

# create a mbatch for a given split_data (mode = train / val / test)
def create_mini_batch_loader(split_data, seed_edge_type, mode):
    # Define seed edges:
    # we pick only a single edge_type to feed edge_label_index (need to verify this approach)

    # neg_sampling in val or test loader causes doubling the edge_label weight producing 2.0 instead of values 1.0 (need to test)
    neg_sampling = ns if mode == 'train' else None
    batch_size =  b if mode == 'train' else (3 * b)
    shuffle = True if mode == 'train' else False

    print(f'mini batch loader for mode {mode}')
    mini_batch_loader = LinkNeighborLoader(
        data=split_data,
        num_neighbors=nn,
        neg_sampling_ratio=neg_sampling, # prev : neg_sampling
        edge_label_index = (seed_edge_type, split_data[seed_edge_type].edge_label_index),
        edge_label=split_data[seed_edge_type].edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return mini_batch_loader

def create(data, model_name):

    if (model_name == 'gcn'):
        # gcn
        model = GCNModel(hidden_channels=dim, data=data, b=b)
    elif (model_name == 'gs'):
        # gs
        model = GSModel(hidden_channels=dim, data = data, b=b)
    elif (model_name == 'gat'):
        # gat
        model = GATModel(hidden_channels=dim, data = data, b=b)
    elif (model_name == 'gin'):
        # gin
        model = GINModel(hidden_channels=dim, data=data, b=b)

    print(model)
    print(f'\nDevice = {device}')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model,optimizer

# learning with batching
def learn_batch(loader, is_directed):
    start = time.time()
    min_loss = 100000000000
    loss_array = []
    val_auc_array = []

    for epoch in range(1, epochs + 1):
        val_auc_array.append(eval_batch(val_loader, is_directed))
        total_loss = 0
        total_examples = 0
        torch.cuda.empty_cache()
        # train for loaders of all edge_types, e.g : train_loader['skill','to','team'], train_loader['member','to','team']
        for seed_edge_type in edge_types:
            print(f'epoch {epoch:03d} : batching for train_loader for seed_edge_type : {seed_edge_type}')
            for sampled_data in loader[seed_edge_type]:
                optimizer.zero_grad()

                sampled_data.to(device)
                pred = model(sampled_data, seed_edge_type, is_directed)
                # The ground_truth and the pred shapes should be 1-dimensional
                # we squeeze them after generation
                if(type(sampled_data) == HeteroData):
                    ground_truth = sampled_data[seed_edge_type].edge_label
                else:
                    ground_truth = sampled_data.edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
        if(epoch % 10 == 0):
            print(f"\n.............Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}.............\n")
        loss_array.append((total_loss / total_examples))

    # plot the figure and save
    fig_output = f'{model_output}/{model_name}.{graph_type}.undir.{agg}.e{epochs}.ns{int(ns)}.b{b}.d{dim}.png'
    xlabel = 'Epochs'
    ylabel = 'Loss'
    title = 'Loss vs Epochs for Embedding Generation'
    plot_graph(torch.arange(1, epochs + 1, 1), loss_array, xlabel, ylabel, title, fig_output)
    fig_output = f'{model_output}/{model_name}.{graph_type}.undir.{agg}.e{epochs}.ns{int(ns)}.b{b}.d{dim}.val_auc_per_epoch.png'
    xlabel = 'Epochs'
    ylabel = 'Val AUC'
    title = 'Validation AUC vs Epochs for Embedding Generation'
    plot_graph(torch.arange(1, epochs + 1, 1), val_auc_array, xlabel, ylabel, title, fig_output)
    print(f'\nit took {(time.time() - start) / 60} mins || {(time.time() - start) / 3600} hours to train the model\n')

# loader can be test or can be validation
@torch.no_grad()
def eval_batch(loader, is_directed):
    preds = []
    ground_truths = []
    for seed_edge_type in edge_types:
        for sampled_data in loader[seed_edge_type]:
            sampled_data.to(device)
            preds.append(model(sampled_data, seed_edge_type, is_directed))
            # ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)
            if (type(sampled_data) == HeteroData):
                # we have ground_truths per edge_label_index
                ground_truths.append(sampled_data[seed_edge_type].edge_label)
                # ground_truth = sampled_data['user','rates','movie'].edge_label
            else:
                ground_truths.append(sampled_data.edge_label)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"AUC: {auc:.4f}\n")
    print(f'................... ending eval...................\n')
    return auc

def addargs(parser):
    parser.add_argument('-domains', nargs = '+', help = 'the domain of the dataset along with the version')
    parser.add_argument('-gnn_models', nargs = '+', help = 'the gnn models to use for embedding generation')
    parser.add_argument('-graph_types', nargs = '+', help = 'the graph types to use')
    parser.add_argument('-agg', nargs = '+', help = 'the aggregation types to use')
    parser.add_argument('-epochs', type = int, help = 'number of epochs for gnn training')
    parser.add_argument('-dim', nargs = '+', type = int, help = 'the embedding dimension to use')
    parser.add_argument('--heads', type = int, required=False, help = 'the number of computational heads to use for gat models')
    parser.add_argument('--use_cpu', type = int, required=False, help = '1 if you want gat to use cpu')

    args = parser.parse_args()
    return args

def set_params(args):

    graph_params.settings['model']['hidden_channels'] = args.dim
    graph_params.settings['model']['gat']['heads'] = args.heads
    graph_params.settings['model']['epochs'] = args.epochs

# sample command including all the args
# cd OpeNTF_Jamil/src/mdl
# python gnn_j.py -domains imdb/title.basics.tsv.filtered.mt75.ts3 -gnn_models gs -epochs 100 -graph_types sm -agg mean -dim 32 64 --heads 8

# similar command in multiline
# python hetero_test.py
# -domains dblp/dblp.v12.json.filtered.mt5.ts2 imdb/title.basics.tsv.filtered.mt5.ts2
# -gnn_models gcn gs gin gat
# -graph_types m sm stm
# -agg none mean
# -dim 32 64 128
# --heads 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN for OpeNTF')
    args = addargs(parser)

    set_params(args)

    epochs = graph_params.settings['model']['epochs']
    hidden_channels = graph_params.settings['model']['hidden_channels'] # will be given as an array
    heads = graph_params.settings['model']['gat']['heads']
    b = graph_params.settings['model']['b']
    lr = graph_params.settings['model']['lr']
    ns = graph_params.settings['model']['negative_sampling']
    nn = graph_params.settings['model']['nn'] # nn = num_neighbors


    # for domain in ['dblp/dblp.v12.json.filtered.mt5.ts2', 'imdb/title.basics.tsv.filtered.mt5.ts2']:
    for domain in args.domains:
    # for domain in ['dblp/dblp.v12.json.filtered.mt5.ts2']:
    # for domain in ['imdb/title.basics.tsv.filtered.mt5.ts2']:
    # for domain in ['uspt/patent.tsv.filtered.mt5.ts2']:
    # for domain in ['gith/data.csv.filtered.mt5.ts2']:
    # for domain in ['dblp/toy.dblp.v12.json']:

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
                    for dim in hidden_channels:
                        if (model_name == 'gcn' and graph_type != 'm'):
                            continue

                        print(f'\nargs : {args}\n')

                        print(f'starting training for \n')
                        print(f'---|||--- {domain}.{model_name}.{graph_type}.undir.{agg}.e{epochs}.b{b}.d{dim} ---|||--- \n')

                        # e.g : domain = 'imdb/title.basics.tsv.filtered.mt5.ts2'
                        filepath = f'../../data/preprocessed/{domain}/gnn/{graph_type}.undir.{agg}.data.pkl'
                        model_output = f'../../data/preprocessed/{domain}/emb'
                        if not os.path.isdir(model_output): os.makedirs(model_output)

                        # logging.info(f'\n-------------------------------------------------------------------------------')
                        # logging.info(f'Model : {model_name} || Graph Type : {graph_type} || Aggregation Type : {agg}')
                        # logging.info(f'-------------------------------------------------------------------------------\n')

                        data = load_data(filepath)
                        is_directed = data.is_directed()

                        train_data, val_data, test_data, edge_types, rev_edge_types = define_splits(data)
                        # validate_splits(train_data, val_data, test_data)

                        ## Sampling for batching > 0
                        if b:
                            train_loader, val_loader, test_loader = {}, {}, {}
                            # create separate loaders for separate seed edge_types
                            for edge_type in edge_types:
                                train_loader[edge_type] = create_mini_batch_loader(train_data, edge_type, 'train')
                                val_loader[edge_type] = create_mini_batch_loader(val_data, edge_type, 'val')
                                test_loader[edge_type] = create_mini_batch_loader(test_data, edge_type, 'test')

                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        print(f"Device: '{device}'")
                        torch.cuda.empty_cache()
                        train_data.to(device)
                        # the train_data is needed to collect info about the metadata
                        model,optimizer = create(train_data, model_name)

                        if b:
                            learn_batch(train_loader, is_directed)
                        else:
                            pass
                            # learn(train_data)
                        torch.cuda.empty_cache()
                        if b:
                            eval_batch(val_loader, is_directed)
                        # store the embeddings
                        with torch.no_grad():
                            for node_type in data.node_types:
                                data[node_type].n_id = torch.arange(data[node_type].x.shape[0])
                            data.to(device)
                            # for simplicity, we just pass seed_edge_type = edge_types[0]. This does not impact any output
                            emb = model(data, edge_types[0], is_directed, emb = True)
                            embedding_output = f'{model_output}/{model_name}.{graph_type}.undir.{agg}.e{epochs}.ns{int(ns)}.b{b}.d{dim}.emb.pt'
                            torch.save(emb, embedding_output, pickle_protocol=4)
                            print(f'\nsaved embedding as : {embedding_output} ..............\n')
                        # eval_batch(test_loader, is_directed)
                        torch.cuda.empty_cache()
