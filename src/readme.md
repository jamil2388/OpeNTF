# Source Code 
This folder contains our implementation of neural team formation, including all the steps of the pipeline from loading the raw datasets, generating sparse matrix, building a neural model, training it, and evaluating it on classification and IR metrics.

1) [``cmn``](./cmn), you can find the abstract class definitions for teams and members as well as inheritance hierarchy for different domains, including:
   1) [``team.py``](./cmn/team.py) class is the parent class for a team 
   2) [``publication.py``](./cmn/publication.py) (research papers as teams and authors as members)
   3) [``movie.py``](./cmn/movie.py) imdb (movies as teams, and cast and crews as members)
   4) [``patent.py``](./cmn/patent.py) (patents as teams and inventors as members)
    
2) [``mdl``](./mdl), we have implemented the neural models here, e.g., [``fnn.py``](./mdl/fnn.py) and a custom dataset class in [``cds.py``](./mdl/cds.py) that will be used to split the data into mini-batches.
3) [``eval``](./eval), we report and plot evaluation of models based on classification metrics and IR metrics here.
4) [``main.py``](./main.py), the main entry point to the benchmark pipeline that realizes all steps related to ``train`` a neural model and plotting the loss, ``test`` on the unseen test set, and ``eval`` based on classification and IR metrics through `polymorphism`.
5) [``param.py``](./param.py), the settings for filtering the data, the number of processes, size of buckets for parallel generation of the sparse matrix, hyperparameters for the model, are all in this file.
6) [``config_prepare_dataset.py``](./config_prepare_dataset.py), the config settings containing hyperparameters related to training graph emebbedings from an edge list.
7) [``model.py``](./model.py), this file contains graph neural network consisting of gin and gcn models.
8) [``datacheck.py``](./datacheck.py), This file is used to generate edges between experts and skills, experts and locations. This file also preprocess the generated edge matrix for expert->skills, and expert->locations to make three separate files for expert nodes, skill nodes, location nodes and on adjacency matrix that includes every edge between these three attributes.
9) [``train_node_emb.py``](./train_node_emb.py), This file is used as a driver script for the GNN defined in model.py to begin training for node embeddings. This file takes the full adjaceny matrix between expert skill and location and returns embeddings for all three types of nodes.
10) [``metapath2vec``](./metapath2vec.py), This file contains two functions. The first one generates random walks based upon metapaths and then the second one trains a word2vec model for the extracted random walks and generate embeddings.
11) [``graph_helper.py``](./graph_helper.py), Helper file consisting of graph functions. It performs read of graphs from an edge list into pytorch geometric object, set data function that creates mini batches and create dataset function that creates Data object of the base graph for Pytorch geometric

# Run

The execution of graph based learning techniques follows the same process and command as for neural network execution.\

List of available graph models: - 
	- bnn_emb_gnn_loc : Graph based embeddging generation involving skills and location followed by bnn model execution.
	- bnn_emb_gnn : Graph based embeddging generation involving followed by bnn model execution.
	- bnn_emb_gnn_loc_meta : Graph based meta paths embedding extraction onvolving skills and location followed by bnn execution.
	- bnn_emb_gnn_meta : Graph based meta paths embedding extraction onvolving skills followed by bnn execution.

For Example: - ``python main.py -data ../data/raw/uspt/toy.patent.tsv -domain uspt -model bnn_emb_gnn -filter 0``
