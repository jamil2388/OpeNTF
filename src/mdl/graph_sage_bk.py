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
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class Model(torch.nn.Module):
    # the data here is the global graph data that we loaded
    # need to verify whether in inductive training we pass the entire data info here
    def __init__(self, hidden_channels, data):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        # self.movie_lin = torch.nn.Linear(20, hidden_channels)
        # this edit is due to the fact that the input size of the linear layer transform is equal
        # to the number of features in x
        self.user_lin = torch.nn.Linear(data['user'].num_features, hidden_channels)
        self.movie_lin = torch.nn.Linear(data['movie'].num_features, hidden_channels)

        # the embeddings of the node_types
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)

        # Instantiate homogeneous gs:
        self.gs = GS(hidden_channels)

        if (type(data) == HeteroData):
            # Convert gs model into a heterogeneous variant:
            self.gs = to_hetero(self.gs, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:

        x_dict = {
            # previously, the lin layer was ignored because of not having any features, we can still ignore it now
            "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].node_id),
            # the feature x conversion part with movie_lin should be revisited
            "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        # pass the embedding of the nodes of all node types and also the edge_index_dict
        # containing the info of all message passing edges
        # this x_dict will now contain the updated embeddings after the message passed layers
        x_dict = self.gs(x_dict, data.edge_index_dict)


        pred = self.classifier(
            x_dict,
            data["user", "rates", "movie"].edge_label_index,
        )

        return pred