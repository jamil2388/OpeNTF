
import torch
import torch_geometric.data
import graph_params
from src.misc import data_handler
from matplotlib import pyplot as plt
import numpy as np

import math
import os

class Graph:

    def __init__(self):
        self.init_variables()

    def init_variables(self):
        self.params = graph_params.settings
        params = self.params

        # model parameters
        model_params = params['model']['m2v']['model_params']
        self.embedding_dim = model_params['embedding_dim']
        self.max_epochs = model_params['max_epochs']
        self.walk_length = model_params['walk_length']
        self.walks_per_node = model_params['walks_per_node']
        self.context_size = model_params['context_size']
        self.num_negative_samples = model_params['num_negative_samples']
        self.batch_size = model_params['batch_size']
        self.shuffle = model_params['shuffle']
        self.num_workers = model_params['num_workers']
        self.lr = model_params['lr']

        self.output_types = params['storage']['output_type']
        self.output_type = self.output_types[0]
        self.domains = list(params['data']['domain'].keys())
        self.domain = self.domains[0]
        self.data_versions = list(params['data']['domain'][self.domain].keys())
        self.data_version = self.data_versions[0]
        self.model_names = list(params['model'].keys())
        self.model_name = self.model_names[5]
        self.graph_edge_types = list(params['model']['m2v']['edge_types'].keys())
        self.graph_edge_type = self.graph_edge_types[0]

        ### Locations ###
        # -------------------------------------------
        # teams_graph_output_folder = f'{base_folder}/{output_type}/{domain}/{data_version}/{model_name}/{graph_edge_type}/'
        # base_filename
        # base_graph_emb_filename
        # base_graph_plot_filename
        self.base_folder = params['storage']['base_folder']
        self.base_filename = params['storage']['base_filename']
        self.teams_graph_output_folder = f'{self.base_folder}/{self.output_type}/{self.domain}/{self.data_version}'
        self.teams_graph_output_filepath = f'{self.teams_graph_output_folder}/{self.base_filename}.{self.model_name}.{self.graph_edge_type}.pkl'
        self.teams_graph_input_filepath = self.teams_graph_output_filepath

        # this is the output folder for preprocessed embeddings and performance data
        self.base_graph_emb_filename = params['storage']['base_graph_emb_filename']
        self.graph_preprocessed_output_folder = f'{self.base_folder}/preprocessed/{self.domain}/{self.data_version}'
        # need to add '{epoch_number}.pkl' while running at each set of epochs
        self.graph_preprocessed_output_filename = f'{self.base_graph_emb_filename}.{self.model_name}.{self.graph_edge_type}'

        # need to add '{epoch_number}.png' while running at each set of epochs
        self.base_graph_plot_filename = params['storage']['base_graph_plot_filename']
        self.graph_plot_filename = f'{self.base_graph_plot_filename}.{self.model_name}.{self.graph_edge_type}.'
        # -------------------------------------------

    # normalizr an np_array into a range of 0-1
    def normalize(self, np_array):
        mx = np.max(np_array)
        mn = np.min(np_array)

        print(f'\nNormalize()\n')
        print(f'Before :')
        print(f'{np_array}')

        np_array = (np_array - mn) / (mx - mn)

        print(f'After :')
        print(f'{np_array}')
        print('')

        return np_array

    # to plot a graph performance
    def plot(self, x, y, output_filepath = None):
        mx = max(y)
        mn = min(y)
        threshold = (mx - mn) // 10

        plt.figure()
        plt.ylabel('Loss')
        plt.ylim(mn - threshold, mx + threshold)
        plt.xlabel('Epochs')
        plt.xlim(0, len(x))
        plt.plot(x, y)
        plt.legend('This is a legend')
        plt.show()

        if(output_filepath):
            print(f'Saving plot in {output_filepath}')
            plt.savefig(output_filepath)

    # for plotting data for multiple sets of epochs or something similar
    def multi_plot(self):
        pass