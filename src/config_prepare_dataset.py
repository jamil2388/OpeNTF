from pathlib import Path
import sys

DATASET_DIR = "../data/preprocessed/uspt/"
# Random Seed
RANDOM_SEED = 42

# Parameters for training node embeddings for base graph
# CONV = "graphsaint_gcn" 
# MINIBATCH = "GraphSaint" # bug
CONV = "gin_gcn"
MINIBATCH = "NeighborSampler"
POSSIBLE_BATCH_SIZES = [256]
POSSIBLE_HIDDEN = [256]
POSSIBLE_OUTPUT = [128]
POSSIBLE_LR = [0.001, 0.0001]
POSSIBLE_WD = [5e-4, 5e-5]
POSSIBLE_DROPOUT = [0.4]
POSSIBLE_NB_SIZE = [-1]
POSSIBLE_NUM_HOPS = [1]
POSSIBLE_WALK_LENGTH = [32]
POSSIBLE_NUM_STEPS = [32]
EPOCHS = 100

# Flags for precomputing similarity metrics
CALCULATE_SHORTEST_PATHS = True # Calculate pairwise shortest paths between all nodes in the graph
CALCULATE_DEGREE_SEQUENCE = True # Create a dictionary containing degrees of the nodes in the graph
CALCULATE_EGO_GRAPHS = True # Calculate the 1-hop ego graph associated with each node in the graph
OVERRIDE = False # Overwrite a similarity file even if it exists
N_PROCESSSES = 4 # Number of cores to use for multi-processsing when precomputing similarity metrics


