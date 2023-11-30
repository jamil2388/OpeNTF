import torch
import torch_geometric.utils
from torch import Tensor
import os
from torch_geometric.data import download_url, extract_zip
import pandas as pd
from torch_geometric.data import HeteroData, Data
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric.transforms as T
import torch.nn.functional as F
from src.mdl.graph_sage import Model
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


def load_data(path):
    movies_path = f'{path}/ml-latest-small/movies.csv'
    ratings_path = f'{path}/ml-latest-small/ratings.csv'

    # if the paths does not exist, we need to download and extract them
    if(not os.path.exists(movies_path) and not os.path.exists(ratings_path)):
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        extract_zip(download_url(url, path), path)

    print('movies.csv:')
    print('===========')
    print(pd.read_csv(movies_path)[["movieId", "genres"]].head())
    print()
    print('ratings.csv:')
    print('============')
    print(pd.read_csv(ratings_path)[["userId", "movieId"]].tail())

    # Load the entire movie data frame into memory:
    movies_df = pd.read_csv(movies_path, index_col='movieId')
    # Load the entire ratings data frame into memory:
    ratings_df = pd.read_csv(ratings_path)

    return movies_df, ratings_df

# return a subset df based on the input col condition
def select_data_subset(df, col = None):

    df_sub = df[df['userId'] < 5]

    # subset df
    print(f'')

    return df_sub

def extract_features(movies_df):
    # Split genres and convert into indicator variables:
    genres = movies_df['genres'].str.get_dummies('|')
    print(genres[["Action", "Adventure", "Drama", "Horror"]].head())

    # Use genres as movie input features:
    movie_feat = torch.from_numpy(genres.values).to(torch.float)

    return movie_feat

def create_unique_mapping(movies_df, ratings_df):
    print(f'unique_user_id before mapping')
    # Create a mapping from unique user indices to range [0, num_user_nodes):
    unique_user_id = ratings_df['userId'].unique()
    print(f'unique_user_id = {unique_user_id.shape}')
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })
    print(f'unique_user_id after mapping')
    print(f'unique_user_id = {unique_user_id.shape}')

    print("Mapping of user IDs to consecutive values:")
    print("==========================================")
    print(unique_user_id.head())
    print()
    # Create a mapping from unique movie indices to range [0, num_movie_nodes):
    unique_movie_id = ratings_df['movieId'].unique()

    print(f'unique_movie_id = {unique_movie_id.shape}')
    unique_movie_id = pd.DataFrame(data={
        'movieId': movies_df.index,
        'mappedID': pd.RangeIndex(len(movies_df)),
    })
    print(f'unique_movie_id after conversion = {unique_movie_id.shape}')
    print("Mapping of movie IDs to consecutive values:")
    print("===========================================")
    print(unique_movie_id.head())

    return unique_user_id, unique_movie_id

# merge to produce ratings_user_id and ratings_movie_id
def merge_indexes(ratings_df, unique_user_id, unique_movie_id):
    # Perform merge to obtain the edges from users and movies:
    ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                               left_on='userId', right_on='userId', how='left')
    ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
    ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                                left_on='movieId', right_on='movieId', how='left')
    ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)

    print(f'ratings_user_id after merge = {ratings_user_id}')
    return ratings_user_id, ratings_movie_id

def create_data(unique_user_id, unique_movie_id, edge_index_user_to_movie):
    data = HeteroData()
    print(f'data.to_dict = {data.to_dict()}')

    # Save node indices:
    data["user"].node_id = torch.arange(len(unique_user_id))
    data["movie"].node_id = torch.arange(len(unique_movie_id))
    print(f'type of data["user"].node_id = {type(data["user"].node_id)}')

    # Add the node features and edge indices:
    data["movie"].x = movie_feat
    data["user", "rates", "movie"].edge_index = edge_index_user_to_movie
    print(f'data.to_dict = {data.to_dict()}')

    # We also need to make sure to add the reverse edges from movies to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)
    print(f'data.to_dict = {data.to_dict()}')

    print(data)

    # assertion checks
    data_assertions(data)

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

    if(type(train_data) == torch_geometric.data.Data):
        # convert to df to compare the overlaps
        train_df = pd.DataFrame(train_data.numpy())
        val_df = pd.DataFrame(val_data.numpy())
        test_df = pd.DataFrame(test_data.numpy())

        train_test_df = pd.merge(train_df, test_df, how = 'inner', on = train_data.columns)

# ignore when you have custom data
def data_assertions(data):
    assert data.node_types == ["user", "movie"]
    assert data.edge_types == [("user", "rates", "movie"),
                               ("movie", "rev_rates", "user")]
    assert data["user"].num_nodes == 610
    assert data["user"].num_features == 0
    assert data["movie"].num_nodes == 9742
    assert data["movie"].num_features == 20
    assert data["user", "rates", "movie"].num_edges == 100836
    assert data["movie", "rev_rates", "user"].num_edges == 100836

def split_assertions(train_data, val_data, test_data):
    assert train_data["user", "rates", "movie"].num_edges == 56469
    assert train_data["user", "rates", "movie"].edge_label_index.size(1) == 24201
    assert train_data["movie", "rev_rates", "user"].num_edges == 56469
    # No negative edges added:
    assert train_data["user", "rates", "movie"].edge_label.min() == 1
    assert train_data["user", "rates", "movie"].edge_label.max() == 1

    assert val_data["user", "rates", "movie"].num_edges == 80670
    assert val_data["user", "rates", "movie"].edge_label_index.size(1) == 30249
    assert val_data["movie", "rev_rates", "user"].num_edges == 80670
    # Negative edges with ratio 2:1:
    assert val_data["user", "rates", "movie"].edge_label.long().bincount().tolist() == [20166, 10083]

def mini_batch_assertions(sampled_data):
    assert sampled_data["user", "rates", "movie"].edge_label_index.size(1) == 3 * 128
    assert sampled_data["user", "rates", "movie"].edge_label.min() == 0
    assert sampled_data["user", "rates", "movie"].edge_label.max() == 1

# define the train, valid, test splits
def define_splits(data):
    # For this, we first split the set of edges into
    # training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly.
    # We can leverage the `RandomLinkSplit()` transform for this from PyG:

    if(type(data) == torch_geometric.data.HeteroData):
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

    print(f'---------------Before Transform-------------')
    print()
    print(f'data["user"] = {data["user"].num_nodes}')
    print(f'data["movie"] = {data["movie"].num_nodes}')
    print(f'data["user","rates","movie"].edge_index = {data["user", "rates", "movie"].edge_index}')
    print(f'data["user","rates","movie"].num_edges = {data["user", "rates", "movie"].num_edges}')
    print(f'data["movie", "rev_rates", "user"].num_edges = {data["movie", "rev_rates", "user"].num_edges}')

    train_data, val_data, test_data = transform(data)

    print()
    print(f'---------------After Transform-------------')

    print("Training data:")
    print("==============")
    print(train_data)
    print(f'........................................')
    # the edges for supervision
    print(f'train_data.edge_label_index and edge_index')
    print(train_data["user", "rates", "movie"].edge_label_index)
    print(train_data["user", "rates", "movie"].edge_index)
    print(f'........................................')
    print()
    print("Validation data:")
    print("================")
    print(f'val_data.edge_label_index and edge_index')
    print(val_data["user", "rates", "movie"].edge_label_index)
    print(val_data["user", "rates", "movie"].edge_index)
    print(f'........................................')
    print()
    print(f'test_data.edge_label_index and edge_index')
    print(test_data["user", "rates", "movie"].edge_label_index)
    print(test_data["user", "rates", "movie"].edge_index)

    print(f'-------------------------------------------')
    print(f'counts')
    print(f'train_data.num_edges = {train_data["user", "rates", "movie"].num_edges}')
    print(f'train_data.reverse.num_edges = {train_data["movie", "rev_rates", "user"].num_edges}')
    print(f'validation_data.num_edges = {val_data["user", "rates", "movie"].num_edges}')
    print(f'test_data.num_edges = {test_data["user", "rates", "movie"].num_edges}')

    return train_data, val_data, test_data

def create_mini_batch_loader(data):
    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.
    # We can make use of the `loader.LinkNeighborLoader` from PyG:

    # Define seed edges:
    edge_label_index = train_data["user", "rates", "movie"].edge_label_index
    edge_label = train_data["user", "rates", "movie"].edge_label
    print(f'edge_label stuffs : {edge_label_index}, {edge_label}')

    mini_batch_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[1, 1],
        neg_sampling_ratio=1.0,
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=3,
        shuffle=True,
    )

    # Inspect a sample:
    # sampled_data = next(iter(train_loader))
    # for i, data in enumerate(train_loader):
    #     print(f'sample data for iteration : {i}')
    #     print(data)
    #     print(f'---------------------------------------\n')

    # print("Sampled mini-batch:")
    # print("===================")
    # print(sampled_data.to_dict().keys())
    # print(f'sampled_data["user"].num_nodes = {sampled_data["user"].num_nodes}')
    # print(f'sampled_data["movie"].num_nodes = {sampled_data["movie"].num_nodes}')
    # print(f'sampled_data["edge1"].num_nodes = {sampled_data["user", "rates", "movie"].num_edges}')
    return mini_batch_loader

def create():

    model = Model(hidden_channels=10, data = train_data)
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model,optimizer

def learn(train_loader):
    for epoch in range(1, 20):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            sampled_data.to(device)
            pred = model(sampled_data)

            ground_truth = sampled_data["user", "rates", "movie"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f'epoch = {epoch}')
        print(f'loss = {loss}')
        print(f'total_examples = {total_examples}')
        print(f'total_loss = {total_loss}')
        # validation part here maybe ?
        auc = eval(val_loader)
        print(f'validation auc = {auc:.4f}')
    return model

def epoch_sampler():
    return

# loader can be test or can be validation
def eval(loader):
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"AUC: {auc:.4f}")
    return auc

if __name__ == '__main__':

    # load data
    movies_df, ratings_df = load_data('../../data/graph/raw/')
    # backup of the main df
    ratings_df_main = ratings_df
    ratings_df = select_data_subset(ratings_df)

    movie_feat = extract_features(movies_df)
    ### assert movie_feat.size() == (9742, 20)  # 20 genres in total.

    unique_user_id, unique_movie_id = create_unique_mapping(movies_df, ratings_df)
    # these dfs will now point to mappedIDs
    ratings_user_id, ratings_movie_id = merge_indexes(ratings_df, unique_user_id, unique_movie_id)

    # With this, we are ready to construct our `edge_index` in COO format
    # following PyG semantics:
    edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
    ### assert edge_index_user_to_movie.size() == (2, 100836)

    # "user", "movie", "user-rates-movie"
    # data = create_data(unique_user_id, unique_movie_id, edge_index_user_to_movie)
    homogeneous_data = create_custom_homogeneous_data()
    heterogeneous_data = create_custom_heterogeneous_data()

    # # draw the graph
    # draw_graph(data)

    # train_data, val_data, test_data = define_splits(homogeneous_data)
    train_data, val_data, test_data = define_splits(heterogeneous_data)
    validate_splits(train_data, val_data, test_data)

    train_loader = create_mini_batch_loader(train_data)
    val_loader = create_mini_batch_loader(val_data)
    test_loader = create_mini_batch_loader(test_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    model,optimizer = create()
    model2 = learn(train_loader)
    assert model.user_emb() == model2.user_emb()
    assert model.movie_emb() == model2.movie_emb()
    eval(test_loader)
