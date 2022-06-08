# from src.cmn.team import Team
import os
from src.cmn.patent import Patent
from src import param
from src.cmn.team_test import Team

settings = param.settings

datapath = '../../data/raw/uspt/toy.patent.tsv'
filter = False

import os
# print(os.listdir('../../data/preprocessed/'))
print(os.listdir(f'../../data/preprocessed/uspt/{os.path.split(datapath)[-1]}'))
print(f'../../data/preprocessed/uspt/{os.path.split(datapath)[-1]}')
# print(f'{os.path.split(datapath)[-1]}')
vecs, indexes = Patent.generate_sparse_vectors(datapath, f'../../data/preprocessed/uspt/{os.path.split(datapath)[-1]}', filter, settings['data'])
print('Reached')


# T1 = Team()
# print()