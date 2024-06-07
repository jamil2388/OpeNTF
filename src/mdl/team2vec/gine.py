import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINEConv


class GINE(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim = 1, train_eps = True)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim = 1, train_eps = True)
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim = 1, train_eps = True)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim = 1, train_eps = True)

    def forward(self, x, edge_index, edge_attr):
        # Node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.conv4(x, edge_index, edge_attr)

        return x


class Classifier(torch.nn.Module):
    def forward(self, source_node_emb, target_node_emb, edge_label_index) -> Tensor:
        # Convert node embeddings to edge-level representations:
        # (e.g. : shapes -> edge_feat_source (4, 16), edge_feat_target (4, 16))
        # the number of rows correspond to total number of labeled_edges in the seed_edge_type, in this case, 4
        edge_feat_source = source_node_emb[edge_label_index[0]]
        edge_feat_target = target_node_emb[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge in the edge_label_index
        return (edge_feat_source * edge_feat_target).sum(dim=-1)


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
                if (data.is_cuda):
                    self.node_lin.append(torch.nn.Linear(data[node_type].num_features, hidden_channels).cuda())
                    self.node_emb.append(torch.nn.Embedding(data[node_type].num_nodes, hidden_channels).cuda())
                else:
                    self.node_lin.append(torch.nn.Linear(data[node_type].num_features, hidden_channels))
                    self.node_emb.append(torch.nn.Embedding(data[node_type].num_nodes, hidden_channels))
        else:
            if (data.is_cuda):
                self.node_lin = torch.nn.Linear(data.num_features, hidden_channels).cuda()
                self.node_emb = torch.nn.Linear(data.num_nodes, hidden_channels).cuda()
            else:
                self.node_lin = torch.nn.Linear(data.num_features, hidden_channels)
                self.node_emb = torch.nn.Linear(data.num_nodes, hidden_channels)

        # Instantiate homogeneous gat:
        self.gine = GINE(hidden_channels)

        if (type(data) == HeteroData):
            # Convert gs model into a heterogeneous variant:
            self.gine = to_hetero(self.gine, metadata=data.metadata())

        # instantiate the predictor class
        self.classifier = Classifier()

    # emb = True enables to just output the embedding generated by the model after
    # the entire training phase has been done
    def forward(self, data, seed_edge_type, is_directed, emb = False) -> Tensor:
        if(type(data) == HeteroData):
            self.x_dict = {}
            for i, node_type in enumerate(data.node_types):
                self.x_dict[node_type] = self.node_lin[i](data[node_type].x) + self.node_emb[i](data[node_type].n_id)
            # `x_dict` holds embedding matrices of all node types
            # `edge_index_dict` holds all edge indices of all edge types
            # edge_attr_dict should hold the similar formatted dict with edge attrs
            self.edge_attr_dict = {}
            for edge_type in data.edge_types:
                self.edge_attr_dict[edge_type] = data[edge_type].edge_attr.view(-1, 1).float()

            self.x_dict = self.gine(self.x_dict, data.edge_index_dict, self.edge_attr_dict)
            if emb : return self.x_dict
        else:
            # for batching mode and homogeneous graphs, this line should be tested by appending node_emb part
            # e.g : if self.b : x_dict['node'] = self.node_lin(data.x) + self.node_emb(data.n_id)
            x = self.node_lin(data.x) + self.node_emb(data.n_id)
            x = self.gine(x, data.edge_index) # need to fix this
            self.x = x
            if emb : return self.x

        # create an empty tensor and concatenate the preds afterwards
        preds = torch.empty(0).to(device = 'cuda' if data.is_cuda else 'cpu')

        if (type(data) == HeteroData):
            # generate predictions only on the edge_label_index of seed_edge_type
            # e.g : for seed_edge_type ['skill','to','team'], we choose to only predict from data['skill','to','team'].edge_label_index
            # source_node_emb contains the embeddings of each node of the defined node_type
            source_node_emb = self.x_dict[seed_edge_type[0]]
            target_node_emb = self.x_dict[seed_edge_type[2]]
            pred = self.classifier(source_node_emb, target_node_emb, data[seed_edge_type].edge_label_index)
            preds = torch.cat((preds, pred.unsqueeze(0)), dim = 1)
        else:
            pred = self.classifier(x, x, data.edge_label_index)
            # pred = self.classifier(x_dict['node'], x_dict['node'], data.edge_label_index)
            preds = torch.cat((preds, pred.unsqueeze(0)), dim = 1)
        return preds.squeeze(dim=0)