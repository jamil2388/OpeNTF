import math

import torch_geometric.data

import graph_params
import src.mdl.graph
from src.misc import data_handler

import os
import torch
from src.mdl import gnn_emb
from torch_geometric.nn import MetaPath2Vec
import numpy as np

class Metapath2Vec(src.mdl.graph.Graph):

    # setup the entire model before running
    def __init__(self):
        # it will inherit all the variables in the Graph class
        super().__init__()
        datapath = self.teams_graph_input_filepath
        self.define_metapath(graph_params.settings['model']['m2v']['metapath'])
        self.load(datapath)

    def define_metapath(self, metapath):
        self.metapath = metapath

    # this will load the desired graph data for running with the model
    def load(self, graph_datapath):
        print(f'graph data to load from : {graph_datapath}')
        self.data = data_handler.load_graph(graph_datapath)

        print(f'loaded graph data : {self.data}')

    # initialize the model
    def init(self):
        assert type(self.data) == torch_geometric.data.hetero_data.HeteroData

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MetaPath2Vec(self.data.edge_index_dict, embedding_dim=self.embedding_dim,
                             metapath=self.metapath, walk_length=self.walk_length, context_size=self.context_size,
                             walks_per_node=self.walks_per_node, num_negative_samples=self.num_negative_samples,
                             sparse=True).to(self.device)

        self.loader = self.model.loader(batch_size = self.batch_size, shuffle = self.shuffle, num_workers = self.num_workers)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr = self.lr)

    # train the model to generate embeddings
    def learn(self, model, optimizer, loader, device, epoch, log_steps = 2, eval_steps = 2000):
        model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # if (i + 1) % log_steps == 0:
            #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
            #            f'Loss: {total_loss / log_steps:.4f}'))
            #     total_loss = 0
        return total_loss / len(loader)

    def run(self, num_epochs):
        self.init()

        losses = []
        list_epochs = []
        min_loss = math.inf

        # this file logs every 10 / 20 epochs
        # it is NOT the final pickle file for embeddings
        with open(f'{self.graph_preprocessed_output_filename}.e{num_epochs}.txt', 'w') as outfile:
            line = f'Graph : \n\n' \
                   f'data = {self.data.__dict__}\n' \
                   f'\nNumber of Epochs : {num_epochs}\n' \
                   f'---------------------------------\n'
            for epoch in range(num_epochs):
                loss = self.learn(self.model, self.optimizer, self.loader, self.device, epoch)
                if(loss < min_loss):
                    min_loss = loss
                    print('.')
                if (epoch % 20 == 0):
                    print(f'Epoch = {epoch : 02d}, loss = {loss : .4f}')

                    # the model() gives all the weights and biases of the model currently
                    # the detach() enables this result to require no gradient
                    # and then we convert the tensor to numpy array
                    weights = self.model.embedding.weight.detach().numpy()
                    weights = self.normalize(weights)
                    weights = np.around(weights, 2)

                    print('\nembedding : \n')
                    print(weights)

                    # lines to write to file
                    line += f'Epoch : {epoch}\n'
                    line += f'--------------------------\n'
                    line += f'Node ----- Embedding -----\n\n'
                    for i, weights_per_node in enumerate(weights):
                        print(weights_per_node)
                        line += f'{i:2} : {weights_per_node}\n'
                    line += f'--------------------------\n\n'
                losses.append(loss)
                list_epochs.append(epoch)
            # write to file
            outfile.write(line)

        self.plot(list_epochs, losses, f'{self.graph_plot_filename}.{num_epochs}.png')
        return list_epochs, losses


def main():
    params = graph_params.settings
    m2v = Metapath2Vec()
    max_epochs = m2v.max_epochs
    graph_preprocessed_output_filename = m2v.graph_preprocessed_output_filename
    print(f'preprocessed embedding output path = {graph_preprocessed_output_filename}')

    for num_epochs in max_epochs:
        list_epochs, losses = m2v.run(num_epochs)



if __name__ == '__main__':
    main()