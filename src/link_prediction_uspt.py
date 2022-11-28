import torch
from torch import Tensor
from torch_geometric.data import download_url, extract_zip
import pandas as pd
import pickle
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import tqdm
import torch.nn.functional as F

def load_pkl():
    path_to_pickle = '../data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3'

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
teamids, skillvecs, membervecs, locvecs = teamsvecs['id'], teamsvecs['skill'], teamsvecs['member'], teamsvecs['loc']

colab = membervecs.T @ skillvecs
print(colab.shape)
print()

data_path = "link_df_75_3_skills.csv"

complete_df = pd.read_csv(data_path)

unique_skills = complete_df['skills'].unique()
unique_skills = pd.DataFrame(data={
    'skills': unique_skills,
    'mappedSkills': pd.RangeIndex(len(unique_skills)),
})

print(unique_skills)
print("=========")
print('Dataframe having complete information')
print(len(complete_df['skills']))
print("=========")
print()

# skills = complete_df['skills'].str.get_dummies('/')
# print(skills.head())
