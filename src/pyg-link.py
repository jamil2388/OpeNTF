import torch
import pickle
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.express as px
import random
import string
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from tqdm import tqdm

# from plotly import graph_objs as go
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import Data, download_url, extract_gz
from torch_geometric.nn import GAE, GCNConv, VGAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges, negative_sampling, degree
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# set the seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


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

link_df = pd.DataFrame(columns=['loc_id', 'loc_name', 'skills'])
#
#
# # for key, value in indexes['i2c'].items():
# #     ex_record = list()
# #     ex_record.append(key)
# #     ex_record.append(value)
# #     for team in teams:
# #         for ix, mem in enumerate(team.members):
# #             if mem.id == value:
# #                 loc = team.members_locations[ix][2]
# #                 ex_record.append(loc)
# #                 break
# #         break
# #     ex_map.append(ex_record)
#
teamids, skillvecs, membervecs, locvecs = teamsvecs['id'], teamsvecs['skill'], teamsvecs['member'], teamsvecs['loc']
# #
colab = locvecs.T @ skillvecs
rows = colab.shape[0]
cols = colab.shape[1]
# print(colab.nonzero())
# rows, cols = colab.nonzero()
# for row in rows:
#     for col in cols:
#         ex_map = list()
for row, col in tqdm(zip(rows, cols), total=len(cols)):
    print('Something')
#
#     ex_map = list()
#     # exit()
#     ex_map.append(row)
#     ex_map.append(indexes['i2l'][row])
#     ex_map.append(col)
#     a_series = pd.Series(ex_map, index=link_df.columns)
#
#     link_df = link_df.append(a_series, ignore_index=True)
#     link_df.append(ex_map)
#
# print(link_df.info())
# print(link_df.drop_duplicates(inplace=True).info())
#
# # link_df.to_csv('loc_df_75_3_skills.csv', index=False)
# exit()
# #
# for ix, team in tqdm(enumerate(teams.keys())):
#     skills = ''.join(list(teams[team].skills))
#     try:
#         for ixm, candidate in enumerate(teams[team].members):
#             series_to_append = list()
#             loc = teams[team].members_details[ixm][2]
#             loc_ix = indexes['l2i'][loc]
#             idname = f'{candidate.id}_{candidate.name}'
#             can_ix = indexes['c2i'][idname]
#             series_to_append.append(loc_ix)
#             series_to_append.append(loc)
#             series_to_append.append(can_ix)
#             series_to_append.append(skills)
#             a_series = pd.Series(series_to_append, index=link_df.columns)
#             link_df = link_df.append(a_series, ignore_index=True)
#     except KeyError:
#         pass
#
# for ix, team in tqdm(enumerate(teams)):
#     skills = ''.join(list(team.skills))
#     try:
#         for ixm, candidate in enumerate(team.members):
#             series_to_append = list()
#             loc = team.members_locations[ixm][2]
#             loc_ix = indexes['l2i'][loc]
#             idname = f'{candidate.id}_{candidate.name}'
#             can_ix = indexes['c2i'][idname]
#             series_to_append.append(loc_ix)
#             series_to_append.append(loc)
#             series_to_append.append(can_ix)
#             series_to_append.append(skills)
#             a_series = pd.Series(series_to_append, index=link_df.columns)
#             link_df = link_df.append(a_series, ignore_index=True)
#     except KeyError:
#         pass
#
#
# # # print(link_df.head())
# link_df.to_csv('link_df_90_7_skills.csv', index=False)

# link_df = pd.read_csv('loc_df_75_3_skills.csv')

# print('head', link_df.head())
# print(link_df.info())
dummy = link_df.drop_duplicates()
print('dummy info', dummy.info())

print('End')