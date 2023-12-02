from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import Tensor
import torch.nn
from torch_geometric.data import HeteroData

class GS(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    # all the node types and edge_label_indices of all edge_types will be passed
    # we have to calculate the predictions based on all combinations
    def forward(self, x_dict, data) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge per edge_label_index:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class Model(torch.nn.Module):
    # the data here is the global graph data that we loaded
    # need to verify whether in inductive training we pass the entire data info here
    def __init__(self, hidden_channels, data):
        super().__init__()
        if(type(data) == HeteroData):
            self.node_lin = []
            self.node_emb = []
            # for each node_type
            node_types = data.node_types
            # linear transformers and node embeddings based on the num_features and num_nodes of the node_types
            # these two are generated such that both of them has the same shape and they can be added together
            for i, node_type in enumerate(node_types):
                self.node_lin.append(torch.nn.Linear(data[node_type].num_features, hidden_channels))
                self.node_emb.append(torch.nn.Embedding(data[node_type].num_nodes, hidden_channels))
        else:
            self.node_lin = torch.nn.Linear(data.num_features, hidden_channels)
            self.node_emb = torch.nn.Linear(data.num_nodes, hidden_channels)

        # Instantiate homogeneous gs:
        self.gs = GS(hidden_channels)

        if (type(data) == HeteroData):
            # Convert gs model into a heterogeneous variant:
            self.gs = to_hetero(self.gs, metadata=data.metadata())

        # instantiate the predictor class
        self.classifier = Classifier()

    def forward(self, data, is_directed) -> Tensor:
        x_dict = {}
        for i, node_type in enumerate(data.node_types):
            x_dict[node_type] = self.node_lin[i](data[node_type].x) + self.node_emb[i](data[node_type].n_id)

        pred = []

        # `x_dict` holds embedding matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gs(x_dict, data.edge_index_dict)
        pred.append(self.classifier(x_dict["source"], x_dict["dest"], data["user", "rates", "movie"].edge_label_index))

        return pred