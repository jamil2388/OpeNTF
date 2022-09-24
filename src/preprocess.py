import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from collections import defaultdict

### read data ###
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

path_to_pickle = '../data/preprocessed/uspt/toy.patent.tsv'

teams_pkl = path_to_pickle + '/teams.pkl'
indexes_pkl = path_to_pickle + '/indexes.pkl'
teamsvecs_pkl = path_to_pickle + '/teamsvecs.pkl'

with open(teams_pkl, 'rb') as tb:
    teams = pickle.load(tb)

with open(indexes_pkl, 'rb') as tb:
    indexes = pickle.load(tb)

with open(teamsvecs_pkl, 'rb') as tb:
    teamsvecs = pickle.load(tb)


all_words_og = [list(teams.values())]
all_words = [val for sublist in all_words_og for val in sublist]

# for i, team in enumerate(all_words):
#     for candidate in team.members:
#         # idname = f'{candidate.id}_{candidate.name}'
#         all_words[i] = candidate.id

# for i in indexes.keys():
# print(indexes)

user_items = dict()

for team in teams.values():
    for candidate in team.members:
        idname = f'{candidate.id}_{candidate.name}'
        expert = indexes['c2i'][idname]
        for entry in candidate.locations:
            loc = entry[2]
            expert_loc = indexes['l2i'][loc]
            if expert_loc not in list(user_items.keys()):
                user_items[expert_loc] = [expert]
            else:
                user_items[expert_loc].append(expert)

for each in user_items.keys():
    user_items[each] = list(set(user_items[each]))

user_count = len(user_items)


np.save("../data/preprocessed/uspt/toy.patent.tsv/user_items.npy", user_items)

traindata = dict(); valdata = dict(); testdata = dict()
for k, v in user_items.items():
    trainlist = random.sample(user_items[k], k=round(len(user_items[k]) * 0.8))
    rem = list(set(user_items[k]) - set(trainlist))
    vallist = random.sample(rem, k=round(len(rem) * 0.5))
    testlist = list(set(rem) - set(vallist))
    traindata[k] = trainlist
    valdata[k] = vallist
    testdata[k] = testlist


np.save("../data/preprocessed/uspt/toy.patent.tsv/traindata.npy", traindata)
np.save("../data/preprocessed/uspt/toy.patent.tsv/valdata.npy", valdata)
np.save("../data/preprocessed/uspt/toy.patent.tsv/testdata.npy", testdata)


def get_train_adj_matrix(train_rating):
    '''
    get adjacent matrix of traindata#
    '''
    item_user_train = defaultdict(set)
    for key in train_rating.keys():
        for i in train_rating[key]:
            item_user_train[i].add(key)
    A_indexs = []
    A_values = []
    for x in train_rating.keys():
        len_u = len(train_rating[x])
        for i in train_rating[x]:
            y = i + user_count
            len_v = len(item_user_train[i])
            A_indexs.append([x,y])
            A_values.append(1/len_u)
            A_indexs.append([y,x])
            #A_values.append(1/len_u)
            A_values.append(1/len_v)
    return A_indexs, A_values


A_indexs, A_values = get_train_adj_matrix(traindata)

dummy_array = teamsvecs['skill'].toarray()

print('teamsvecs', teamsvecs['skill'].toarray()[0])
# print('indexes for s2i:', indexes['s2i'])
# print('indexes for i2s:', indexes['i2s'])

np.save("../data/preprocessed/uspt/toy.patent.tsv/A_indexs.npy", A_indexs)
np.save("../data/preprocessed/uspt/toy.patent.tsv/A_values.npy", A_values)
