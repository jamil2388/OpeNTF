import torch
import torch.nn as nn

class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.update_fn = nn.Linear(input_dim, output_dim)
        self.aggregate_fn = nn.Linear(input_dim, output_dim)
    def forward(self, h, adj_matrix):
        messages = torch.matmul(adj_matrix, h)  # Aggregating messages from neighbors
        aggregated = self.aggregate_fn(messages)  # Applying the aggregate function
        updated = self.update_fn(h) + aggregated  # Updating the node embeddings
        return updated
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))
        self.layers.append(GNNLayer(hidden_dim, output_dim))
    def forward(self, features, adj_matrix):
        h = features.clone()
        for layer in self.layers:
            h = layer(h, adj_matrix)
        return h

if __name__ == '__main__':
    # Example usage
    input_dim = 16
    hidden_dim = 32
    output_dim = 8
    num_layers = 3
    # Generating random node features and adjacency matrix
    num_nodes = 10
    features = torch.randn(num_nodes, input_dim)
    adj_matrix = torch.randn(num_nodes, num_nodes)
    # Creating a GNN model
    gnn = GNN(input_dim, hidden_dim, output_dim, num_layers)
    # Forward pass through the GNN
    embeddings = gnn(features, adj_matrix)
    print("Node embeddings:")
    print(embeddings)