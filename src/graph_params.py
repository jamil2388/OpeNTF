'''
this file contains the parameters to do all the graph based tasks

1) reading teamsvecs pickle data
2) creating graph data
3) loading graph data
4) generating embeddings

'''

settings = {
    'model':{
            'gnn':{},
            'gcn':{},
            'gan':{},
            'gin':{},
            'n2v':{},
            'm2v':{
                'metapath' : [
                    ('member','to','id'),
                    ('id', 'to', 'skill'),
                    ('skill','to','id'),
                    ('id', 'to', 'member'),
                ],
                'edge_types' : {
                    'STE' : {},
                    'SE' : {},
                    'STE_TL' : {},
                    'STEL' : {},
                },
                'model_params':{
                    'max_epochs' : [200],
                    'embedding_dim' : 5,
                    'walk_length' : 6,
                    'context_size' : 3,
                    'walks_per_node' : 5,
                    'num_negative_samples' : 5,
                    'batch_size' : 5,
                    'shuffle' : True,
                    'num_workers' : 1,
                    'lr' : 0.01,
                }
            }
        },
    'data':{
        'domain': {
            'dblp':{
                'toy.dblp.v12.json':{},
            },
            'uspt':{
                'toy.patent.tsv':{},
            },
            'imdb':{
                'toy.title.basics.tsv':{},
            },
        },
    },
    'storage':{
        'base_folder' : '../../data/graph',
        'output_type': [
            'raw',
            'preprocessed'
        ],
        'base_filename' : 'teamsvecs',
        'base_graph_emb_filename' : 'teamsvecs.emb',
        'base_graph_plot_filename' : 'teamsplot',
    },
    'misc':{
        'graph_datapath' : '../../data/graph/raw/dblp/toy.dblp.v12.json/metapath2vec/STE/teams_graph.pkl',
        'preprocessed_embedding_output_path' : '../../data/graph/preprocessed/dblp/toy.dblp.v12.json/metapath2vec/STE/teamsvecs_emb.pkl',
        'domain' : 'dblp',
        'dataset_version' : 'toy.dblp.v12.json',
        'model' : 'metapath2vec',
        'edge_type' : 'STE',
        'file_name' : 'teams_graph.pkl',
    }
}