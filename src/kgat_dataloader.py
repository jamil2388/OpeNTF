import pickle
import pandas as pd
import os
from time import time
from tqdm import tqdm


class DataLoader:
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as infile: self.teams = pickle.load(infile)
        self.train_dict = {}
        self.generate_info()
        self.write_data()

    def generate_info(self):
        for team in self.teams.values():
            counter = 0
            for candidate in team.members:
                idname = f'{candidate.id}_{candidate.name}'
                if idname in self.train_dict.keys():
                    self.train_dict[idname].append(team.members_details[counter])
                else:
                    self.train_dict[idname] = []
                    self.train_dict[idname].append(team.members_details[counter])
                counter += 1

    def write_data(self):
        with open("../data/preprocessed/uspt/toy.patent.tsv/data.txt", 'w') as f:
            for key, value in self.train_dict.items():
                f.write('%s %s\n' % (key, value))





