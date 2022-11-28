import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
import pickle
import tqdm
import pandas as pd
import ast
import numpy as np
import numpy as np
from scipy import sparse
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

demo_df = pd.read_csv('uspt_graph_toy.csv')

skills_matrix = torch.load('skills_matrix_toy.pt')

demo_df['skills_matrix'] = skills_matrix

unique_loc_nodes = demo_df['loc_id'].unique()
unique_loc_names = demo_df['loc_name'].unique()
unique_ex_nodes = demo_df['ex_id'].unique()

edge_index_loc_to_expert = torch.stack([torch.from_numpy(demo_df['loc_id'].values.astype('float64')), torch.from_numpy(demo_df['ex_id'].values.astype('float64'))], dim=0)

print('Unique Location Nodes: ', unique_loc_nodes)
print('Total Number of Unique Location Nodes: ', len(unique_loc_nodes))
print()
print('Unique Locations: ', unique_loc_names)
print('Total Number of Unique Locations: ', len(unique_loc_names))
print()
print('Unique Expert Nodes: ', unique_ex_nodes)
print('Total Number of Experts: ', len(unique_ex_nodes))
print()
print('Edge Index Loc to Expert: ', edge_index_loc_to_expert)
print('Shape of Tensor for Edge Index: ', edge_index_loc_to_expert.size())

data = HeteroData()

# Save node indices:
data["location"].node_id = torch.Tensor(unique_loc_nodes.astype('float32')).type(torch.int64)
    # .type(torch.LongTensor)
data["experts"].node_id = torch.Tensor(demo_df['ex_id'].astype('float32')).type(torch.int64)
    # .type(torch.LongTensor)
# data["location"].node_id = torch.Tensor(unique_loc_nodes.astype('float64'))
# data["experts"].node_id = torch.Tensor(unique_ex_nodes.astype('float64'))

# Add the node features and edge indices:
data["experts"].x = skills_matrix.type(torch.float32)
data["location", "of", "experts"].edge_index = edge_index_loc_to_expert.type(torch.int64)

# For Message Passing in both the directions
transform = T.Compose([T.ToUndirected()])
data = transform(data)

print(data)

transform = T.RandomLinkSplit(
    num_val=0.10,
    num_test=0.10,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("location", "of", "experts"),
    rev_edge_types=("experts", "rev_of", "location"),
)

print('Data Before Splitting', data)
print("==============")
print()
train_data, val_data, test_data = transform(data)
print("Training data:")
print("==============")
print(train_data)
print()
print("Validation data:")
print("================")
print(val_data)


edge_label_index = train_data["location", "of", "experts"].edge_label_index
edge_label = train_data["location", "of", "experts"].edge_label

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("location", "of", "experts"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True
)

# Inspect a sample:
sampled_data = next(iter(train_loader))

print("Sampled mini-batch:")
print("===================")
print(sampled_data)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(111, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["location"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["experts"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "location": self.user_emb(data["location"].node_id),
            "experts": self.movie_lin(data["experts"].x) + self.movie_emb(data["experts"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["location"],
            x_dict["experts"],
            data["location", "of", "experts"].edge_label_index,
        )

        return pred


model = Model(hidden_channels=64)

print(model)
# model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 6):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        pred = model.forward(sampled_data)
        loss = F.binary_cross_entropy_with_logits(pred, sampled_data["location", "of", "experts"].edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

edge_label_index = val_data["location", "of", "experts"].edge_label_index
edge_label = val_data["location", "of", "experts"].edge_label

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("location", "of", "experts"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)

sampled_data = next(iter(val_loader))

print("Sampled mini-batch:")
print("===================")
print(sampled_data)

preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        # TODO: Collect predictions and ground-truths and write them into
        # `preds` and `ground_truths`.
        preds.append(model.forward(sampled_data))
        ground_truths.append(sampled_data["location", "of", "experts"].edge_label)
        # raise NotImplementedError

pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")