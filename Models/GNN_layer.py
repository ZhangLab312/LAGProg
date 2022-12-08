import torch.nn
# from dgl.nn.pytorch import SumPooling
# from dgl.nn.pytorch.conv import GraphConv
from torch_geometric.nn import GraphConv, GCNConv, GATConv, SAGEConv, TAGConv, SGConv
import torch.nn.functional as F


class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, feature_len, dim):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = feature_len
        self.dim = dim
        self.gnn_layers = torch.nn.ModuleList([])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(self.n_layer):
            self.gnn_layers.append(GraphConv(in_channels=self.feature_len if i == 0 else dim,
                                             out_channels=dim))

        self.factor = None

    def forward(self, x, edge_index):

        edge_index = edge_index.to(self.device)
        for index, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if index != self.n_layer - 1:
                x = torch.relu(x)
        graph_embedding = x
        return graph_embedding
