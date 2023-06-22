from stellargraph.data import UniformRandomMetaPathWalk
import pandas as pd
import numpy as np
import multiprocessing
from gensim.models import Word2Vec
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph import StellarGraph
import multiprocessing
import stellargraph as sg
import networkx as nx

from joblib import Parallel,delayed
from tqdm import tqdm
import pickle
import os
from sklearn.preprocessing import normalize
import time
import contextlib
import joblib
from tqdm import tqdm
from gensim.models import KeyedVectors

walk_length = 64
num_walks_per_node = 40
emb_dim = 128  # Need to Change
emb_fpath = 'mp2v_{}_{}_{}.npy'.format(emb_dim, num_walks_per_node, walk_length)

metapaths = [
    ["experts", "skills", "experts"],
    ["experts", "skills", "experts", "skills", "experts"],
    ["experts", "skills", "loc", "skills", "experts"],
]

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def generate_random_walks(graph_obj, num_walks_per_node, walk_length, metapaths):
    random_walk_object = UniformRandomMetaPathWalk(graph_obj)
    cpu_count = multiprocessing.cpu_count()
    list_nodes = list(graph_obj.nodes())

    num_chunks = max(1, len(list_nodes) // cpu_count)
    chunk_len = (len(list_nodes) // num_chunks)
    chunks = [list_nodes[i * chunk_len: (i + 1) * chunk_len] for i in range(num_chunks)]

    with tqdm(desc='Progress:', total=len(chunks)) as progress_bar:
        all_walks = Parallel(n_jobs=cpu_count)(
            delayed(aux_gen_walks)(
                node_chunk, walk_length, random_walk_object, metapaths, num_walks_per_node
            )
            for node_chunk in chunks
            if len(node_chunk) > 0  # Exclude empty chunks
        )
        progress_bar.update(len(chunks))

    return [walk for walks in all_walks for walk in walks]

def aux_gen_walks(node_chunk, walk_length, random_walk_object, metapaths, num_walks=1):
    walks = []
    for node in node_chunk:
        walks.extend(random_walk_object.run(nodes=[node], length=walk_length, n=num_walks, metapaths=metapaths))
    return walks

def find_missing_nodes(lst):
    # Create a set of all numbers from the minimum to the maximum in the list
    all_numbers = set(range(min(lst), max(lst) + 1))

    # Find the difference between the set of all numbers and the list
    missing_numbers = list(all_numbers - set(lst))

    return missing_numbers

def generate_walks(sk_nodes, ex_nodes, loc_nodes, df_edge_list):
    start_time = time.time()
    sk_nodes.rename(columns={'col1': 'sk_id'}, inplace=True)
    ex_nodes.rename(columns={'col1': 'ex_id'}, inplace=True)
    loc_nodes.rename(columns={'col1': 'loc_id'}, inplace=True)

    ex_nodes_unique = pd.DataFrame(index=ex_nodes['ex_id'].unique())
    sk_nodes_unique = pd.DataFrame(index=sk_nodes['sk_id'].unique())
    loc_nodes_unique = pd.DataFrame(index=loc_nodes['loc_id'].unique())

    graph_obj = StellarGraph({
        "experts": ex_nodes_unique,
        "skills":sk_nodes_unique,
        "loc": loc_nodes_unique
    },
        df_edge_list
    )

    print(graph_obj.info())

    print(len(graph_obj.edges()))
    print(np.mean(list(graph_obj.node_degrees().values())))

    walks_save_file = "mp2v_random_walks_{}_{}.npy".format(walk_length, num_walks_per_node)
    try:
        walks_np_arr = np.load(walks_save_file, allow_pickle=True)
        walks = [ list(_) for _ in walks_np_arr]

        print("Number of random walks: {}".format(len(walks)))

        unique_nodes = set()
        for walk in walks:
            unique_nodes.update(walk)
        print("Number of unique nodes in the random walks: {}".format(len(unique_nodes)))

        str_walks = [[str(n) for n in walk] for walk in walks]

    except FileNotFoundError:
        print('Starting to Generate Random Walks...')
        walks = generate_random_walks(graph_obj, num_walks_per_node, walk_length, metapaths)

        print("Number of random walks: {}".format(len(walks)))

        with open(walks_save_file, 'wb') as file:
            pickle.dump(walks, file)

        unique_nodes = set()
        for walk in walks:
            unique_nodes.update(walk)
        print("Number of unique nodes in the random walks: {}".format(len(unique_nodes)))

        str_walks = [[str(n) for n in walk] for walk in walks]
        print("Time Taken To Generate Random Walks %s seconds ---" % (time.time() - start_time))
    return str_walks, graph_obj

def generate_metapaths(str_walks, graph_obj):

    start_time = time.time()

    try:
        normalized_dict = np.load(emb_fpath, allow_pickle=True).item()
        print('Node Embeddings File Exists, Loading the file : ')
    except FileNotFoundError:
        print('File Does not exists, Generating Embeddings...')
        print('Starting to Train Word2Vec Model')
        word2vec_params = {
            'sg': 0,
            "vector_size": emb_dim,
            "alpha": 0.5,
            "min_alpha": 0.001,
            'window': 5,
            'min_count': 0,
            "workers": multiprocessing.cpu_count(),
            "negative": 1,
            "hs": 0,  # 0: negative sampling, 1:hierarchical  softmax
            'compute_loss': True,
            'epochs': 10,
            'cbow_mean': 1,
        }

        iters = 100
        mp2v_model = Word2Vec(**word2vec_params)
        mp2v_model.build_vocab(str_walks)
        losses = []
        learning_rate = 0.5
        step_size = (0.5 - 0.001) / iters

        for i in tqdm(range(iters)):
            trained_word_count, raw_word_count = mp2v_model.train(
                str_walks,
                compute_loss=True,
                start_alpha=learning_rate,
                end_alpha=learning_rate,
                total_examples=mp2v_model.corpus_count,
                epochs=1
            )
            loss = mp2v_model.get_latest_training_loss()
            losses.append(loss)
            print('>> ', i, ' Loss:: ', loss, learning_rate)
            learning_rate -= step_size

        # mp2v_model.wv.save('vectors.kv')
        # ======== Save node weights ============ #
        print(mp2v_model.wv)
        node_embeddings = []
        count = 0
        node_embeddings = mp2v_model.wv.vectors

        # Create a dictionary mapping node IDs to embeddings
        node_embedding_dict = {node: embedding for node, embedding in zip(graph_obj.nodes(), node_embeddings)}

        for key, value in node_embedding_dict.items():
            reshaped_array = value.reshape((1, 128))
            node_embedding_dict[key] = reshaped_array

        average_array = np.mean(list(node_embedding_dict.values()), axis=0)
        for i, key in enumerate(range(83310, 83381)):
            node_embedding_dict[key] = np.copy(average_array)

        print(len(node_embedding_dict))

        np.save(emb_fpath, node_embedding_dict)
        print("Time Taken To Train W2V Model %s seconds ---" % (time.time() - start_time))

        reshaped_dict = {key: value.reshape((1, 128)) for key, value in node_embedding_dict.items()}

        mean_array = np.mean(np.array(list(reshaped_dict.values())), axis=0)


        for i in find_missing_nodes(list(reshaped_dict.keys())):
            reshaped_dict.update({i: mean_array})

        print('Output:', len(reshaped_dict))

        node_embedding_dict = {'experts': {}, 'skills': {}, 'loc': {}}
        for node in graph_obj.nodes(node_type="experts"):
            node_embedding_dict['experts'][node] = reshaped_dict[node]

        for node in graph_obj.nodes(node_type="skills"):
            node_embedding_dict['skills'][node] = reshaped_dict[node]

        for node in graph_obj.nodes(node_type="loc"):
            node_embedding_dict['loc'][node] = reshaped_dict[node]

        vstack_embs = np.vstack((list(node_embedding_dict['experts'].values()), list(node_embedding_dict['skills'].values()),
                                 list(node_embedding_dict['loc'].values())))

        vstack_array = np.squeeze(vstack_embs, axis=1)
        normalized_array = normalize(vstack_array, axis=1, norm='l1')

        ex_nodes = [node for node in graph_obj.nodes() if graph_obj.node_type(node) == "experts"]
        ex_len = len(ex_nodes)
        sk_nodes = [node for node in graph_obj.nodes() if graph_obj.node_type(node) == "skills"]
        loc_nodes = [node for node in graph_obj.nodes() if graph_obj.node_type(node) == "loc"]
        sk_len = len(sk_nodes)
        loc_len = len(loc_nodes)

        normalized_dict = {'experts': {}, 'skills': {}, 'loc': {}}
        normalized_dict['experts'] = normalized_array[:ex_len]
        normalized_dict['skills'] = normalized_array[ex_len:ex_len + sk_len]
        normalized_dict['loc'] = normalized_array[ex_len + sk_len:]

        print(normalized_dict.keys())
        print(normalized_dict['experts'].shape)
        print(normalized_dict['skills'].shape)
        print(normalized_dict['loc'].shape)

        with open('metapaths_emb.pkl', 'wb') as file:
            pickle.dump(normalized_dict, file)

        return  normalized_dict