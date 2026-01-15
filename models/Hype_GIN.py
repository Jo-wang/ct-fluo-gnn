from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GINConv

class GINMultiGraphModelwoSol(nn.Module):
    def __init__(self, input_dim_a, input_dim_c, embedding_size=128, hl_size=64, n_gnn_layers=3, n_hl=2, pooling="add", activation="ReLU", dropout=0.5):
        super(GINMultiGraphModelwoSol, self).__init__()


        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError("Invalid activation function")

  
        if pooling == "add":
            self.pooling = global_add_pool
        elif pooling == "mean":
            self.pooling = global_mean_pool
        elif pooling == "max":
            self.pooling = global_max_pool
        else:
            raise ValueError("Invalid pooling method")

        self.dropout = dropout

   
        self.gnn_layers = nn.ModuleList()
        input_size = input_dim_a
        for _ in range(n_gnn_layers):
            self.gnn_layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(input_size, hl_size),
                        nn.BatchNorm1d(hl_size),
                        self.activation,
                        nn.Linear(hl_size, hl_size),
                        self.activation
                    )
                )
            )
            input_size = hl_size

        # Embedding
        self.embedding_acid = nn.Embedding(2, embedding_size)
        self.embedding_concentration = nn.Embedding(2, embedding_size)

    
        fc_input_size = hl_size * n_gnn_layers + embedding_size * 2
        self.fc_layers = nn.ModuleList()
        for _ in range(n_hl - 1):
            self.fc_layers.append(nn.Linear(fc_input_size, hl_size))
            self.fc_layers.append(self.activation)
            self.fc_layers.append(nn.Dropout(p=self.dropout))
            fc_input_size = hl_size
        self.fc_layers.append(nn.Linear(fc_input_size, 2))

    def forward(self, data_a, data_c, concentration):
        x_a, edge_index_a = data_a.x, data_a.edge_index
        gnn_outputs = []

        for conv in self.gnn_layers:
            x_a = conv(x_a, edge_index_a)
            x_a = F.dropout(x_a, p=self.dropout, training=self.training)
            gnn_outputs.append(self.pooling(x_a, data_a.batch))

        x_a_ = torch.cat(gnn_outputs, dim=1)

        indices = torch.tensor([0 if smi == 'C' else 1 for smi in data_c.smiles], dtype=torch.long, device=data_c.x.device)
        x_c_ = self.embedding_acid(indices)    

        ind_conc = torch.where(concentration == 0.5, 0, 1).long().to(concentration.device)
        conc_encoded = self.embedding_concentration(ind_conc)

        x = torch.cat((x_a_, x_c_, conc_encoded), dim=1)

        for layer in self.fc_layers:
            x = layer(x)
        x = torch.sigmoid(x)
        return x
