import pickle
import torch
import argparse


def addargs(parser):
    parser.add_argument('--data')
    parser.add_argument('--model')
    parser.add_argument('--emb_filepath')
    parser.add_argument('--teamsvecs_filepath')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser('filetest.py')
    args = addargs(parser)

    print(f'given arguments : {args}')

    if(args.teamsvecs_filepath):
        with open(args.teamsvecs_filepath, 'rb') as f:
            print(f'\nloading {args.teamvsvecs}')
            teamsvecs = pickle.load(f)
            print(f'teamsvecs file : \n{teamsvecs}\n')

    if(args.emb_filepath is not None):
        print(f'\nloading {args.emb_filepath}')
        emb = torch.load(args.emb_filepath, map_location=torch.device('cpu'))
        print(f'emb file : \n{emb}\n')
