# from src.cmn.team import Team
import os
from patent import Patent
from src import param
#from src.cmn.team_test import Team
from src import fnn_main
# import src.main
import numpy as np
import json
from json import JSONEncoder
from sklearn.model_selection import KFold, train_test_split
from multiprocessing import freeze_support
settings = param.settings

datapath = '../../data/raw/uspt/toy.patent.tsv'
# datapath = '../../data/raw/uspt/patent.tsv'
filter = False


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def create_evaluation_splits(n_sample, n_folds, train_ratio=0.85, output='./'):
    train, test = train_test_split(np.arange(n_sample), train_size=train_ratio, random_state=0, shuffle=True)
    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()

    skf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
        splits['folds'][k] = dict()
        splits['folds'][k]['train'] = train[trainIdx]
        splits['folds'][k]['valid'] = train[validIdx]

    with open(f'{output}/splits.json', 'w') as f:
        json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)

    return splits

prep_output = f'../../data/preprocessed/uspt/{os.path.split(datapath)[-1]}'
print(prep_output)
output = f'../../output/'
# exit()
import os

if __name__ == '__main__':
    freeze_support()
    # print(os.listdir('../../data/preprocessed/'))
    # print(os.listdir(f'../../data/preprocessed/uspt/{os.path.split(datapath)[-1]}'))
    # print(f'../../data/preprocessed/uspt/{os.path.split(datapath)[-1]}')
    # print(f'{os.path.split(datapath)[-1]}')
    vecs, indexes = Patent.generate_sparse_vectors(datapath, f'../../data/preprocessed/uspt/{os.path.split(datapath)[-1]}', filter, settings['data'])
    del vecs['skill']
    vecs['skill'] = vecs['loc']
    del vecs['loc']
    # print(vecs)
    # exit()
    print('Reached. Starting fnn')
    splits = create_evaluation_splits(len(indexes['t2i']), 3, 0.85, output=prep_output)
    print(splits)
    fnn_main.main(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}test/fnn', settings['model']['baseline']['fnn'], settings['model']['cmd'])


    # T1 = Team()
    # print()