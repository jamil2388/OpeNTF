import pickle
import pandas as pd
import os
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, teams_file, teamsvecs_file, indexes_file):
        with open(teams_file, 'rb') as infile: self.teams = pickle.load(infile)
        with open(teamsvecs_file, 'rb') as infile: self.teamsvecs = pickle.load(infile)
        with open(indexes_file, 'rb') as infile: self.indexes = pickle.load(infile)
        self.train_dict = {}
        self.generate_data()
        self.write_data()
        self.split_data()
        # self.print_info()

    def generate_data(self):
        for team in self.teams.values():
            mem_list = []
            loc_list = []
            for candidate in team.members:
                idname = f'{candidate.id}_{candidate.name}'
                print('idname:', idname, 'index of member:', self.indexes['c2i'][idname])
                mem_list.append(self.indexes['c2i'][idname])
            for loc in team.members_details:
                print('Country ', loc[2], 'index of location:', self.indexes['l2i'][loc[2]])
                loc_list.append(self.indexes['l2i'][loc[2]])
            self.train_dict.update({mem_list[i]: loc_list[i] for i in range(len(mem_list))})
        print(self.train_dict)

    def write_data(self):
        with open("../data/preprocessed/uspt/toy.patent.tsv/data.txt", 'w') as f:
            for key, value in self.train_dict.items():
                f.write('%s %s\n' % (key, value))

    def split_data(self):
        data_df = pd.DataFrame(self.train_dict.items())
        print(data_df)
        train, test = train_test_split(data_df, test_size=0.2, random_state=2019)
        train.to_csv('../data/preprocessed/uspt/toy.patent.tsv/train.txt', header=None, index=None, sep=' ', mode='a')
        test.to_csv('../data/preprocessed/uspt/toy.patent.tsv/test.txt', header=None, index=None, sep=' ', mode='a')
