from torch_geometric.data import download_url, extract_zip
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec

from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear


data = HeteroData()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def load_pkl():
    path_to_pickle = '../data/preprocessed/uspt/patent.tsv.filtered.mt90.ts7'

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

#het link
# two nodes = loc, expert
# expert = skill

# expert_csv = pd.DataFrame(columns=['expert_id', 'expert_name', 'expert_skills'])
# loc_expert_csv = pd.DataFrame(columns=['loc_id', 'expert_id', 'rating'])
i = 0; j = 0

expert_csv = pd.read_csv('expert_csv.csv')
loc_expert_csv = pd.read_csv('loc_expert_csv.csv')

# Code to Generate files
# for team in tqdm(teams):
#     skills_concat = str()
#     for skill in team.skills:
#         skills_concat = skills_concat + skill + '|'
#     skills_concat = skills_concat[:-1]
#     for v, candidate in enumerate(team.members):
#         loc = team.members_locations[v][2]
#         try:
#             loc_id = indexes['l2i'][loc]
#             expert_id = candidate.id
#             expert_name = candidate.name
#             if expert_id not in expert_csv['expert_id'].values:
#                 expert_csv.loc[i] = [expert_id, expert_name, skills_concat]
#                 i = i+1
#             else:
#                 ix = expert_csv[expert_csv['expert_id'] == expert_id].index.values
#                 expert_csv.loc[ix, 'expert_skills'] = expert_csv.loc[ix, 'expert_skills'] + '|' + skills_concat
#             loc_expert_csv.loc[j] = [loc_id, expert_id, 1]
#             j = j+1
#         except KeyError:
#             pass
#
# # loc_expert_csv['rating'] = 1
# expert_csv.to_csv('expert_csv.csv', index=False)
# loc_expert_csv.to_csv('loc_expert_csv.csv', index=False)
#


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class GenresEncoder(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}
        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


expert_x, expert_mapping = load_node_csv(
    'expert_csv.csv', index_col='expert_id', encoders={
        'expert_name': SequenceEncoder(),
        'expert_skills': GenresEncoder()
    })

loc_y, loc_mapping = load_node_csv(
    'loc_expert_csv.csv', index_col='loc_id', encoders={
        'expert_id': SequenceEncoder()
    })

# _, loc_mapping = load_node_csv('loc_expert_csv.csv', index_col='loc_id')

# data = HeteroData()


# data['loc'].x = loc_y
# data['loc'].num_nodes = len(loc_mapping)# Users do not have any features.
#
# data['expert'].x = expert_x
# data['expert'].num_nodes = len(expert_mapping)


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


edge_index, edge_label = load_edge_csv(
    'loc_expert_csv.csv',
    src_index_col='loc_id',
    src_mapping=loc_mapping,
    dst_index_col='expert_id',
    dst_mapping=expert_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

data = HeteroData({'expert': {'x':expert_x}, 'loc':{'x':loc_y}},
                  loc__has__expert={'edge_index':edge_index, 'edge_label':edge_label})

# data['loc', 'has', 'expert'].edge_index = edge_index
# data['loc', 'has', 'expert'].edge_label = edge_label

# homo_data = data.to_homogeneous()

# print('data to homo: ', homo_data)
# print(data.to_dict())

# data = T.ToUndirected()(data)
# data = T.AddSelfLoops()(data)
# data = T.NormalizeFeatures()(data)
data = T.RandomNodeSplit()(data)


print(data)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = GNN(hidden_channels=64, out_channels=24)
model = to_hetero(model, data.metadata(), aggr='sum')


train_input_nodes = ('expert', data['expert'].train_mask)

train_loader = NeighborLoader(data, num_neighbors=[10] *2, shuffle=True, input_nodes=train_input_nodes)

for t in train_loader:
    print(t)
    break

exit()

# 3rd Way
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # ('loc', 'has', 'expert'): GCNConv(-1, hidden_channels)
                ('loc', 'has', 'expert'): SAGEConv((-1, -1), hidden_channels)
                # ('paper', 'rev_writes', 'author'): GATConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['loc'])

#
# model = HeteroGNN(hidden_channels=64, out_channels=24,
#                   num_layers=2)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

exit()


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = GNN(hidden_channels=64, out_channels=24)
# model = to_hetero(model, data.metadata(), aggr='sum')

# model.train()
data = T.ToSparseTensor()(data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['loc'].train_mask
    loss = F.cross_entropy(out['loc'][mask], data['loc'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

# print(model)
# print(train())
exit()

data_x = Data(
        x=data.x_dict['loc'],
        edge_index=data.edge_index_dict[('loc', 'has', 'expert')],
        y=data.y_dict['expert'])

print(data_x)

#SAGEConv
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


## Metapath2Vec
metapath = [
    ('loc', 'has', 'expert'),
    ('expert', 'from', 'loc')
]

model = MetaPath2Vec(data.edge_index_dict,
                     embedding_dim=128,
                     metapath=metapath,
                     walk_length=1,
                     context_size=1,
                     walks_per_node=3,
                     num_negative_samples=1,
                     sparse=True)

loader = model.loader(batch_size=32, shuffle=True)

for idx, (pos_rw, neg_rw) in enumerate(loader):
    if idx == 10: break
    print(idx, pos_rw.shape, neg_rw.shape)

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


# def train(epoch, log_steps=500, eval_steps=1000):
#     model.train()
#
#     total_loss = 0
#     for i, (pos_rw, neg_rw) in enumerate(loader):
#         optimizer.zero_grad()
#         loss = model.loss(pos_rw.to(device), neg_rw.to(device))
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         if (i + 1) % log_steps == 0:
#             print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
#                    f'Loss: {total_loss / log_steps:.4f}'))
#             total_loss = 0
#
#         if (i + 1) % eval_steps == 0:
#             acc = test()
#             print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
#                    f'Acc: {acc:.4f}'))


@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model('author', batch=data.y_index_dict['author'])
    y = data.y_dict['author']

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm],
                      y[test_perm], max_iter=150)


# if __name__ == '__main__':

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = GNN(hidden_channels=64, out_channels=24)


train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[15] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=('loc', data['loc'].train_mask),
)

batch = next(iter(train_loader))


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to('cuda:0')
        batch_size = batch['loc'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out['loc'][:batch_size],
                               batch['loc'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


for epoch in range(1, 2):
    print(train(epoch))
    # acc = test()
    # print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')





# url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# extract_zip(download_url(url, '.'), '.')
#
# movie_path = './ml-latest-small/movies.csv'
# rating_path = './ml-latest-small/ratings.csv'
#
# print(pd.read_csv(movie_path).head())
# print(pd.read_csv(rating_path).head())
#
#
# def load_node_csv(path, index_col, encoders=None, **kwargs):
#     df = pd.read_csv(path, index_col=index_col, **kwargs)
#     mapping = {index: i for i, index in enumerate(df.index.unique())}
#
#     x = None
#     if encoders is not None:
#         xs = [encoder(df[col]) for col, encoder in encoders.items()]
#         x = torch.cat(xs, dim=-1)
#
#     return x, mapping
#
# class SequenceEncoder(object):
#     def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
#         self.device = device
#         self.model = SentenceTransformer(model_name, device=device)
#
#     @torch.no_grad()
#     def __call__(self, df):
#         x = self.model.encode(df.values, show_progress_bar=True,
#                               convert_to_tensor=True, device=self.device)
#         return x.cpu()
#
# class GenresEncoder(object):
#     def __init__(self, sep='|'):
#         self.sep = sep
#
#     def __call__(self, df):
#         genres = set(g for col in df.values for g in col.split(self.sep))
#         mapping = {genre: i for i, genre in enumerate(genres)}
#
#         x = torch.zeros(len(df), len(mapping))
#         for i, col in enumerate(df.values):
#             for genre in col.split(self.sep):
#                 x[i, mapping[genre]] = 1
#         return x
#
#
# movie_x, movie_mapping = load_node_csv(
#     movie_path, index_col='movieId', encoders={
#         'title': SequenceEncoder(),
#         'genres': GenresEncoder()
#     })
#
# _, user_mapping = load_node_csv(rating_path, index_col='userId')
