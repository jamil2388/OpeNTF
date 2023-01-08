import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
import pickle
import tqdm
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear
import torch.nn.functional as F


def load_pkl():
    # path_to_pickle = '../data/preprocessed/uspt/toy.patent.tsv'
    path_to_pickle = '../data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3'

    teams_pkl = path_to_pickle + '/teams.pkl'
    indexes_pkl = path_to_pickle + '/indexes.pkl'
    teamsvecs_pkl = path_to_pickle + '/teamsvecs.pkl'

    with open(teams_pkl, 'rb') as tb:
        teams = pickle.load(tb)

    with open(indexes_pkl, 'rb') as tb:
        indexes = pickle.load(tb)

    # with open(teamsvecs_pkl, 'rb') as tb:
    #     teamsvecs = pickle.load(tb)

    return teams, indexes


teams, indexes = load_pkl()

# skill_nodes = pd.DataFrame(indexes['i2s'].items(), columns=['sk_id', 'skills'])
# loc_nodes = pd.DataFrame(indexes['i2l'].items(), columns=['loc_id', 'locations'])
# expert_nodes = pd.DataFrame(indexes['i2c'].items(), columns=['ex_id', 'experts'])
# skill_nodes.to_csv('skill_nodes.csv', index=False)
# loc_nodes.to_csv('loc_nodes.csv', index=False)
# expert_nodes.to_csv('ex_nodes.csv', index=False)

skill_nodes = pd.read_csv('skill_nodes.csv')
loc_nodes = pd.read_csv('loc_nodes.csv')
expert_nodes = pd.read_csv('ex_nodes.csv')

skill_list = list(skill_nodes['skills'])
loc_list = list(loc_nodes['locations'])

ex_to_sk = dict(); ex_to_loc = dict()
# for team in teams:
#     locs = [x[2] for x in teams[team].members_details]
#     exs = [f'{mem.id}_{mem.name}'for mem in teams[team].members]
#     ex_to_loc.update(dict(zip(exs, locs)))
#     for mem in teams[team].members:
#         ex_to_sk[f'{mem.id}_{mem.name}'] = list(teams[team].skills)

# print(ex_to_sk)
# print()
# print(ex_to_loc)

# matched_locid = list(); matched_exid_loc = list(); matched_skid = list(); matched_exid = list()
# for key in tqdm.tqdm(ex_to_loc.keys()):
#     loc = ex_to_loc[key]
#     if loc in loc_list:
#         matched_locid.append(loc_nodes[loc_nodes['locations'] == loc]['loc_id'].iloc[0])
#         matched_exid_loc.append(expert_nodes[expert_nodes['experts'] == key]['ex_id'].iloc[0])
#
# for key in tqdm.tqdm(ex_to_sk.keys()):
#     for skill in ex_to_sk[key]:
#         if skill in skill_list:
#             matched_skid.append(skill_nodes[skill_nodes['skills'] == skill]['sk_id'].iloc[0])
#             matched_exid.append(expert_nodes[expert_nodes['experts'] == key]['ex_id'].iloc[0])

# edge_loc2ex = pd.DataFrame({'ex_id': matched_exid_loc, 'loc_id': matched_locid}, columns=['ex_id', 'loc_id'])
# edge_loc2ex.to_csv('edge_loc2ex.csv', index=False)
edge_loc2ex = pd.read_csv('edge_loc2ex.csv')
# edge_sk2ex = pd.DataFrame({'ex_id': matched_exid, 'sk_id': matched_skid}, columns=['ex_id', 'sk_id'])
# edge_sk2ex.to_csv('edge_sk2ex.csv', index=False)
edge_sk2ex = pd.read_csv('edge_sk2ex.csv')


expert_nodes = pd.read_csv('ex_nodes.csv', usecols=['ex_id'], dtype={'ex_id': np.int64})
skills_nodes = pd.read_csv('skill_nodes.csv', usecols=['sk_id'], dtype={'sk_id': np.int64})
loc_nodes = pd.read_csv('loc_nodes.csv', usecols=['loc_id'], dtype={'loc_id': np.int64})
sk_to_ex = pd.read_csv('edge_sk2ex.csv', usecols=['ex_id', 'sk_id'], dtype={'ex_id': np.int64, 'sk_id': np.int64})
ex_to_loc = pd.read_csv('edge_loc2ex.csv', usecols=['ex_id', 'loc_id'], dtype={'ex_id': np.int64, 'loc_id': np.int64})
print('Number of Expert Nodes', len(expert_nodes))
print('Number of Skill Nodes', len(skills_nodes))
print('Number of Location Nodes', len(loc_nodes))
print('Number of Skill to Expert Edges', len(sk_to_ex))
print('Number of Location to Expert Edges', len(ex_to_loc))

# demo_df = pd.read_csv('uspt_graph_toy.csv')
#
# skills_matrix = torch.load('skills_matrix_toy.pt')

# skill_matrices = [t.numpy() for t in list(skills_matrix)]
# skills_df = pd.DataFrame({'skills_matrices': skill_matrices}, columns=['skills_mapped', 'skills_matrices'])
#
# offset = max(max(list(demo_df['loc_id']), list(demo_df['ex_id']))) + 1
# # print(offset)
# # exit()
# mapping = [i + offset for i, _ in enumerate(skills_df['skills_matrices'])]
# demo_df['skills_mapped'] = mapping
# skills_df['skills_mapped'] = mapping

# node_mapping = [src_mapping[index] for index in demo_df['skills_matrix']]
# demo_df[]

# print(torch.from_numpy(demo_df['ex_id'].values.astype('float64')).size())


# unique_loc_nodes = demo_df['loc_id'].unique()
# unique_loc_names = demo_df['loc_name'].unique()
# unique_ex_nodes = demo_df['ex_id'].unique()
#
# edge_index_loc_to_expert = torch.stack([torch.from_numpy(demo_df['loc_id'].values.astype('float64')), torch.from_numpy(demo_df['ex_id'].values.astype('float64'))], dim=0)
# edge_index_expert_to_skills = torch.stack([torch.from_numpy(demo_df['ex_id'].values.astype('float64')), torch.from_numpy(demo_df['skills_mapped'].values)], dim=0)
#
# print('Unique Location Nodes: ', unique_loc_nodes)
# print('Total Number of Unique Location Nodes: ', len(unique_loc_nodes))
# print()
# print('Unique Locations: ', unique_loc_names)
# print('Total Number of Unique Locations: ', len(unique_loc_names))
# print()
# print('Unique Expert Nodes: ', unique_ex_nodes)
# print('Total Number of Experts: ', len(unique_ex_nodes))
# print()
# print('Edge Index Loc to Expert: ', edge_index_loc_to_expert)
# print('Shape of Tensor for Edge Index: ', edge_index_loc_to_expert.size())
# print()
# print('Edge Index Expert to Skills: ', edge_index_expert_to_skills)
# print('Edge Index Expert to Skills: ', edge_index_expert_to_skills.size())

unique_loc_nodes = loc_nodes['loc_id'].unique()
unique_sk_nodes = skills_nodes['sk_id'].unique()
unique_ex_nodes = expert_nodes['ex_id'].unique()

edge_index_loc_to_expert = torch.stack([torch.from_numpy(ex_to_loc['loc_id'].values.astype('int64')), torch.from_numpy(ex_to_loc['ex_id'].values.astype('int64'))], dim=0)
edge_index_sk_to_expert = torch.stack([torch.from_numpy(sk_to_ex['sk_id'].values.astype('int64')), torch.from_numpy(sk_to_ex['ex_id'].values.astype('int64'))], dim=0)

print('Unique Location Nodes: ', unique_loc_nodes)
print('Total Number of Unique Locations: ', len(unique_loc_nodes))
print()
print('Unique Skill Nodes: ', unique_sk_nodes)
print('Total Number of Unique Skills: ', len(unique_sk_nodes))
print()
print('Unique Expert Nodes: ', unique_ex_nodes)
print('Total Number of Experts: ', len(unique_ex_nodes))
print()
print('Edge Index Loc to Expert: ', edge_index_loc_to_expert)
print('Shape of Tensor for Edge Index: ', edge_index_loc_to_expert.size())
print('Edge Index Skill to Expert: ', edge_index_sk_to_expert)
print('Shape of Tensor for Edge Index: ', edge_index_sk_to_expert.size())

data = HeteroData()

# Save node indices:

data["location"].node_id = torch.Tensor(unique_loc_nodes.astype('float64')).type(torch.LongTensor)
data["experts"].node_id = torch.Tensor(unique_ex_nodes.astype('float64')).type(torch.LongTensor)
data["skills"].node_id = torch.Tensor(unique_sk_nodes.astype('float64')).type(torch.LongTensor)
# data["location"].node_id = torch.Tensor(unique_loc_nodes.astype('float64'))
# data["experts"].node_id = torch.Tensor(unique_ex_nodes.astype('float64'))

# Add the node features and edge indices:
# data["experts"].x = skills_matrix.type(torch.int8).type(torch.LongTensor)
data["location", "of", "experts"].edge_index = edge_index_loc_to_expert.type(torch.int64).type(torch.LongTensor)
data["skills", "of", "experts"].edge_index = edge_index_sk_to_expert.type(torch.int64).type(torch.LongTensor)


# For Message Passing in both the directions
transform = T.Compose([T.ToUndirected()])
data = transform(data)

print(data)

# data["location"].node_id = torch.Tensor(unique_loc_nodes.astype('float32')).type(torch.int64)
#     # .type(torch.LongTensor)
# data["experts"].node_id = torch.Tensor(demo_df['ex_id'].astype('float32')).type(torch.int64)
#
# data["skills"].node_id = skills_matrix.type(torch.int64)
#
#     # .type(torch.LongTensor)
# # data["location"].node_id = torch.Tensor(unique_loc_nodes.astype('float64'))
# # data["experts"].node_id = torch.Tensor(unique_ex_nodes.astype('float64'))
#
# # Add the node features and edge indices:
# # data["experts"].x = skills_matrix.type(torch.float32)
# data["location", "of", "experts"].edge_index = edge_index_loc_to_expert.type(torch.int64)
# data["experts", "have", "skills"].edge_index = edge_index_expert_to_skills.type(torch.int64)

# exit()

# torch.save(data, 'uspt_sk_loc_ex_data.pt')


# transform = T.RandomLinkSplit(
#     num_val=0.05,
#     num_test=0.1,
#     disjoint_train_ratio=0.3,
#     neg_sampling_ratio=2.0,
#     add_negative_train_samples=False,
#     edge_types=("location", "of", "experts"),
#     rev_edge_types=("experts", "rev_of", "location"),
# )

train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[("location", "of", "experts")],
    rev_edge_types=[("experts", "rev_of", "location")],
)(data)

print('Data Before Splitting', data)
print("==============")
print()
# train_data, val_data, test_data = transform(data)
print("Training data:")
print("==============")
print(train_data)
print()
print("Validation data:")
print("================")
print(val_data)

print(data.validate())

# edge_label_index = train_data["location", "of", "experts"].edge_label_index
# edge_label = train_data["location", "of", "experts"].edge_label
#
# train_loader = LinkNeighborLoader(
#     data=train_data,
#     num_neighbors=[20, 10],
#     neg_sampling_ratio=2.0,
#     # edge_label_index=(("location", "of", "experts"), edge_label_index),
#     # edge_label=edge_label,
#     batch_size=128,
#     shuffle=True
# )
#
# # Inspect a sample:
# sampled_data = next(iter(train_loader))
#
# print("Sampled mini-batch:")
# print("===================")
# print(sampled_data)
# exit()

target_edge = ['location', 'experts']

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


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[target_edge[0]][row], z_dict[target_edge[1]][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        # self.movie_lin = torch.nn.Linear(20, hidden_channels)
        # self.loc_emb = torch.nn.Embedding(train_data["location"].num_nodes, hidden_channels)
        # self.ex_emb = torch.nn.Embedding(train_data["experts"].num_nodes, hidden_channels)
        # self.sk_emb = torch.nn.Embedding(train_data["skills"].num_nodes, hidden_channels)

        self.emb_dict = torch.nn.ModuleDict({
            key: torch.nn.Embedding(node_id.numel(), hidden_channels)
            for key, node_id in data.node_id_dict.items()
        })

        self.encoder = GNN(hidden_channels)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='mean')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, train_data, edge_index_dict, edge_label_index):
        x_dict = {key: emb.weight for key, emb in self.emb_dict.items()}
        z_dict = self.encoder(x_dict, edge_index_dict)
        # return z_dict
        return self.decoder(z_dict, edge_label_index)

    # def return_embeddings(self, x_dict, edge_index_dict, edge_label_index):
    #     z_dict = self.encoder(x_dict, edge_index_dict)
    #     return z_dict

    def return_embeddings(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = {key: emb.weight for key, emb in self.emb_dict.items()}
        z_dict = self.encoder(x_dict, edge_index_dict)
        return z_dict


model = Model(hidden_channels=64)


print(model)
# model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

##### New Train #####

# with torch.no_grad():
#     model.encoder(train_data.x_dict, train_data.edge_index_dict)

# def train():
total_loss = 0
total_examples = 0
for epoch in range(1, 11):
    model.train()  # sets the model in training mode
    optimizer.zero_grad()

    pred = model(train_data, train_data.edge_index_dict,
                 train_data[
                     "location", "of", "experts"].edge_label_index)  # The notation model(something) is calling the forward function
    # Add here The computations for the embeddings

    embdict = model.return_embeddings(train_data, train_data.edge_index_dict,
                 train_data[
                     "location", "of", "experts"].edge_label_index)

    # embdict = model.return_embeddings(data["location"].x, train_data.edge_index_dict)
    locc = embdict['location']
    skillss = embdict['skills']

    # End here the computations for the embeddings

    # target = data[target_edge].edge_label
    loss = torch.nn.BCEWithLogitsLoss()(pred, train_data['location', 'of', 'experts'].edge_label)
    loss.backward()
    optimizer.step()
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
    # print(f"Epoch: {epoch:03d}")
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

print(locc)

# for team in teams:


    # return (float(loss), embdict)


##### New Train Ends #####


# with open('graph_toy.pkl', 'wb') as fb:
#     pickle.dump(preds[-1], fb)

# print(preds[-1]['location'].shape)

# for t in teams:
#     for sk in t.skills:
#

# exit()
# with open('graph_sk_loc_ex.pkl', 'wb') as fb:
#     pickle.dump(preds[-1], fb)
#
# exit()


# for epoch in range(1, 6):
#     total_loss = total_examples = 0
#     for sampled_data in tqdm.tqdm(train_loader):
#         optimizer.zero_grad()
#         pred = model.forward(sampled_data)
#         loss = F.binary_cross_entropy_with_logits(pred, sampled_data["location", "of", "experts"].edge_label)
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * pred.numel()
#         total_examples += pred.numel()
#     print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

edge_label_index = val_data["location", "of", "experts"].edge_label_index
edge_label = val_data["location", "of", "experts"].edge_label

# val_loader = LinkNeighborLoader(
#     data=val_data,
#     num_neighbors=[20, 10],
#     edge_label_index=(("location", "of", "experts"), edge_label_index),
#     edge_label=edge_label,
#     batch_size=3 * 128,
#     shuffle=False,
# )
#
# sampled_data = next(iter(val_loader))
#
# print("Sampled mini-batch:")
# print("===================")
# print(sampled_data)

preds = []
ground_truths = []
# for sampled_data in tqdm.tqdm(val_loader):
with torch.no_grad():
    # TODO: Collect predictions and ground-truths and write them into
    # `preds` and `ground_truths`.
    preds.append(model.forward(val_data, val_data.edge_index_dict,
                 val_data[
                     "location", "of", "experts"].edge_label_index))
    ground_truths.append(val_data["location", "of", "experts"].edge_label)
        # raise NotImplementedError

pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = 0
try:
    auc = roc_auc_score(ground_truth, pred)
except ValueError:
    pass
# auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")