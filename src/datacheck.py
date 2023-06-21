import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import lil_matrix, vstack, csr_matrix, hstack
import pickle
import random
from tqdm import tqdm


def preprocess(sk_to_ex, ex_to_loc):
    print('Starting to Preprocess')
    ex_to_loc = pd.read_csv(ex_to_loc, header=None, names=['ex_id', 'loc_id'])
    # ex_to_loc.rename(columns={'ex': 'ex_id', 'syn_loc': 'syn_loc'}, inplace=True)
    sk_to_ex = pd.read_csv(sk_to_ex, header=None, names=['ex_id', 'sk_id'])

    mapping_ex = dict();
    mapping_sk = dict();
    mapping_loc = dict()
    c = 0
    for ix, i in tqdm(enumerate(list(sk_to_ex['ex_id'].values)), total=len(list(sk_to_ex['ex_id'].values))):
        try:
            if i not in mapping_ex.keys():
                mapping_ex[i] = c
                c += 1
            else:
                pass
        except IndexError:
            pass

    for ix, i in tqdm(enumerate(list(ex_to_loc['ex_id'].values)), total=len(list(ex_to_loc['ex_id'].values))):
        try:
            if i not in mapping_ex.keys():
                mapping_ex[i] = c
                c += 1
            else:
                pass
        except IndexError:
            pass

    for ix, i in tqdm(enumerate(list(sk_to_ex['sk_id'].values)), total=len(list(sk_to_ex['sk_id'].values))):
        try:
            if i not in mapping_sk.keys():
                mapping_sk[i] = c
                c += 1
            else:
                pass
        except IndexError:
            pass

    for ix, i in tqdm(enumerate(list(ex_to_loc['loc_id'].values)), total=len(list(ex_to_loc['loc_id'].values))):
        try:
            if i not in mapping_loc.keys():
                mapping_loc[i] = c
                c += 1
            else:
                pass
        except IndexError:
            pass

    # Precompute mapping dictionaries outside the loop
    mapped_ex_ids = {k: int(v) for k, v in mapping_ex.items()}
    mapped_sk_ids = {k: int(v) for k, v in mapping_sk.items()}

    # Create boolean masks for matching conditions
    ex_id_mask = sk_to_ex['ex_id'].isin(mapped_ex_ids.keys())
    sk_id_mask = sk_to_ex['sk_id'].isin(mapped_sk_ids.keys())

    # Update 'Mapped_ex' and 'Mapped_sk' columns using boolean indexing
    sk_to_ex.loc[ex_id_mask, 'Mapped_ex'] = sk_to_ex.loc[ex_id_mask, 'ex_id'].map(mapped_ex_ids)
    sk_to_ex.loc[sk_id_mask, 'Mapped_sk'] = sk_to_ex.loc[sk_id_mask, 'sk_id'].map(mapped_sk_ids)

    # Precompute mapping dictionaries outside the loop
    mapped_ex_ids = {k: int(v) for k, v in mapping_ex.items()}
    mapped_loc_ids = {k: int(v) for k, v in mapping_loc.items()}

    # Create boolean masks for matching conditions
    ex_id_mask = ex_to_loc['ex_id'].isin(mapped_ex_ids.keys())
    loc_id_mask = ex_to_loc['loc_id'].isin(mapped_loc_ids.keys())

    # Update 'Mapped_ex' and 'Mapped_loc' columns using boolean indexing
    ex_to_loc.loc[ex_id_mask, 'Mapped_ex'] = ex_to_loc.loc[ex_id_mask, 'ex_id'].map(mapped_ex_ids)
    ex_to_loc.loc[loc_id_mask, 'Mapped_loc'] = ex_to_loc.loc[loc_id_mask, 'loc_id'].map(mapped_loc_ids)

    sk_to_ex = sk_to_ex.astype({'Mapped_ex': int, 'Mapped_sk': int})
    ex_to_loc = ex_to_loc.astype({'Mapped_ex': int, 'Mapped_loc': int})

    sk_to_ex.to_csv('sk_to_ex_mapped.csv')
    ex_to_loc.to_csv('ex_to_loc_mapped.csv')

    ex_edges = pd.DataFrame(sk_to_ex[['Mapped_ex', 'Mapped_sk']])
    el_edges = pd.DataFrame(ex_to_loc[['Mapped_ex', 'Mapped_loc']])

    ex_edges.columns = ['source', 'target']
    el_edges.columns = ['source', 'target']

    ex_edges.to_csv('ex_edges.csv', index=False)
    el_edges.to_csv('el_edges.csv', index=False)

    fl = list(sk_to_ex['Mapped_ex'].values) + list(ex_to_loc['Mapped_ex'].values)
    sl = list(sk_to_ex['Mapped_sk'].values) + list(ex_to_loc['Mapped_loc'].values)

    expert_nodes = pd.DataFrame({'col1': fl})
    expert_nodes.to_csv('expert_nodes.csv', header=False, index=False)

    skill_nodes = pd.DataFrame({'col1': list(sk_to_ex['Mapped_sk'].values)})
    skill_nodes.to_csv('skill_nodes.csv', header=False, index=False)

    loc_nodes = pd.DataFrame({'col1': list(ex_to_loc['Mapped_loc'].values)})
    loc_nodes.to_csv('loc_nodes.csv', header=False, index=False)

    df_edgelist = pd.DataFrame({'source': fl, 'target': sl})

    df_edgelist.to_csv('df_edge_list.txt', sep=' ', header=True, index=False)

    return expert_nodes, skill_nodes, loc_nodes, df_edgelist


def edge_generation(vecs):
    mem_to_loc = pd.DataFrame(columns=[0, 1])
    print('Processing for loc')
    for ix in tqdm(range(vecs['member'].shape[0])):
        for mx in list(vecs['member'][ix].nonzero()[1]):
            x = list(vecs['loc'][ix].nonzero()[1])
            l3 = [(mx, x1) for x1 in x]
            ldf = pd.DataFrame(l3)

            mem_to_loc = pd.concat([mem_to_loc, ldf]).reset_index(drop=True)

    mem_to_loc.to_csv('mem_to_loc.csv', header=False, index=False)

    mem_to_sk = pd.DataFrame(columns=[0, 1])

    print('processing for sk')
    for ix in tqdm(range(vecs['member'].shape[0])):
        for mx in list(vecs['member'][ix].nonzero()[1]):
            l2 = list(vecs['skill'][ix].nonzero()[1])
            l3 = [(mx, x) for x in l2]
            ldf = pd.DataFrame(l3)

            mem_to_sk = pd.concat([mem_to_sk, ldf]).reset_index(drop=True)
    mem_to_sk.to_csv('mem_to_sk.csv', header=False, index=False)

    return mem_to_sk, mem_to_loc


def emb_process(gnn_path, meta_file):
    emb_file = torch.load(gnn_path)
    # with open('metapaths_emb.pkl', 'rb') as file:
    #     meta_file = pickle.load(file)
    mem_emb = emb_file[:vecs['member'].shape[1]]
    skill_emb = emb_file[vecs['member'].shape[1]:mem_emb.shape[0] + vecs['skill'].shape[1]]
    print('Skill emb before processing', skill_emb.shape)
    loc_emb = emb_file[mem_emb.shape[0] + skill_emb.shape[0]:]
    loc_emb = vstack([loc_emb, loc_emb[-1:]])
    loc_emb = torch.from_numpy(loc_emb.todense())
    print('Loc emb before processing', loc_emb.shape)
    mem_emb_meta = meta_file['experts']
    skill_emb_meta = meta_file['skills']
    loc_emb_meta = np.concatenate((meta_file['loc'], meta_file['loc'][-1].reshape((1, 128))), axis=0)
    emb_dict_meta = dict(); emb_dict = dict()
    if os.path.exists('emb_dict_meta.pkl'):
        with open('emb_dict_meta.pkl', 'rb') as file:
            emb_dict_meta = pickle.load(file)
    else:
        print('Processing for Skill Meta')
        emb_meta_s = {}
        for i in tqdm.tqdm(range(vecs['skill'].shape[0])):
            x = skill_emb_meta[[list(vecs['skill'][i].nonzero()[1])]]
            x = csr_matrix(x.mean(axis=1, dtype=float))

            emb_meta_s[i] = x
        emb_dict_meta['skill'] = vstack(list(emb_meta_s.values()))
        print(emb_dict_meta['skill'].shape)

        print('Processing for Loc Meta')
        emb_meta_l = {}

        for i in tqdm(range(vecs['loc'].shape[0])):
            f = loc_emb_meta[[list(vecs['loc'][i].nonzero()[1])]]
            f = csr_matrix(f.mean(axis=1, dtype=float))
            emb_meta_l[i] = f
        emb_dict_meta['loc'] = vstack(list(emb_meta_l.values()))
        print(emb_dict_meta['loc'].shape)

        with open('emb_dict_meta.pkl', 'wb') as file:
            pickle.dump(emb_dict_meta, file)

    if os.path.exists('emb_dict_gnn.pkl'):
        with open('emb_dict_gnn.pkl', 'rb') as file:
            emb_dict = pickle.load(file)
    else:
        print('Processing for Skill GNN')
        emb_s = {}
        for i in tqdm.tqdm(range(vecs['skill'].shape[0])):
            g = skill_emb.numpy()[[list(vecs['skill'][i].nonzero()[1])]]
            g = csr_matrix(g.mean(axis=1, dtype=float))

            if g.shape[0] == 1:
                pass
            else:
                print(g.shape)
                exit()
            emb_s[i] = g
        emb_dict['skill'] = vstack(list(emb_s.values()))
        print('Skill Shape Before hstack', emb_dict['skill'].shape)

        print('Processing for Loc GNN')
        emb_l = {}
        for i in tqdm.tqdm(range(vecs['loc'].shape[0])):
            j = loc_emb.numpy()[[list(vecs['loc'][i].nonzero()[1])]]

            j = csr_matrix(j.mean(axis=1, dtype=float))

            emb_l[i] = j
        emb_dict['loc'] = vstack(list(emb_l.values()))
        print('Loc Shape Before hstack', emb_dict['loc'].shape)

        with open('emb_dict_gnn.pkl', 'wb') as file:
            pickle.dump(emb_dict, file)

        return emb_dict, emb_dict_meta


def plot_skew(x, y):
    plt.figure()

    fig = plt.figure(figsize= (8, 6))

    plt.bar(x, y, width=0.6)
    plt.xlabel('Locations')
    plt.ylabel('Number of Experts')
    plt.title('Skewness of Experts in Locations')
    plt.savefig('Skewness of Experts in Locations')
    plt.show()


def add_synthetic(df):
    tcounts_loc = len(df)
    tloc = int(tcounts_loc/len(set(df['loc'])))
    all_locs = dict.fromkeys(set(df['loc']), 0)
    mem_tosynloc = pd.DataFrame(columns=[0, 1])
    tdict = dict()
    for row in tqdm(df.itertuples(), total=len(df)):
        k, v = random.choice(list(all_locs.items()))
        if v != tloc:

            df.loc[row.Index, 'syn_loc'] = k
            all_locs[k] += 1
            # v += 1
        else:
            df.loc[row.Index, 'syn_loc'] = k

    df.to_csv('mem_synloc.csv', index=False)
    df = df.astype({'syn_loc': int})

    print('Synthetic Data Generated')

    return df

    # Commented to avoid generation of synthetic data
    # df = pd.read_csv('mem_to_loc.csv', header=None, names=['ex', 'loc'])
    # df = df.groupby(['loc']).size().reset_index(name='counts')

    # df_syn = add_synthetic(df)
    #
    # expert_nodes, skill_nodes, loc_nodes, EX_edges, EL_edges, edge_list = preprocess('mem_to_sk.csv', df_syn)

