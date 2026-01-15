import copy
import torch
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from dataset_aug import MoleculesDataset
from torch_geometric.data import DataLoader as PyGDataLoader
from models.Hype_GIN import GINMultiGraphModelwoSol

criterion = torch.nn.MSELoss()
mae = torch.nn.L1Loss(reduction="sum")


embedding_size = 97 #107
hl_size = 352 #214
n_gnn_layers = 4 #3
n_hl = 3
pooling = "max"
activation = "leaky_ReLU"
learning_rate = 5.4477152960668946e-05
bs = 3
seed = 1
dropout = 0.016036098461465188


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

df = pd.read_csv("data/dataset.csv", na_values=["None"], header=None, skiprows=1)
df.columns = ["x", "y", "con", "flu", "levis"]
df["levis"] = df["levis"].fillna("C").astype(str)
df_train, df_test = train_test_split(df, test_size=0.1, random_state=seed)
# 5-Fold 
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

avg_test_loss = 0
fold_test_losses = [] 
best_val_loss = float("inf")
best_epoch = 0
best_model_state = None
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
    df_train_fold, df_val_fold = df_train.iloc[train_idx], df_train.iloc[val_idx]
    train_dataset = MoleculesDataset(df_train_fold)
    val_dataset = MoleculesDataset(df_val_fold)
    train_loader = PyGDataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=bs, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GINMultiGraphModelwoSol(
        input_dim_a=train_dataset[0][0].num_node_features,
        input_dim_c=train_dataset[0][1].num_node_features if train_dataset[0][1] is not None else 2,
        embedding_size=embedding_size,
        hl_size=hl_size,
        n_gnn_layers=n_gnn_layers,
        n_hl=n_hl,
        pooling=pooling,
        activation=activation,
        dropout=dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(5000):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            data1, data3, con, y = data
            data1, data3 = data1.to(device), data3.to(device)
            con, y = con.to(device), y.to(device)
            output = model(data1, data3, con)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data1, data3, con, y = data
                data1, data3 = data1.to(device), data3.to(device)
                con, y = con.to(device), y.to(device)
                output = model(data1, data3, con)
                loss = mae(output, y)
                val_loss += loss.item()
        val_loss /= len(val_dataset)
        if epoch == 0:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())  
        if epoch - best_epoch > 1000:  
            break
    torch.save(best_model_state, f"ckpt/model_fold{fold}.pth")
    fold_test_losses.append(best_val_loss)
    
best_fold = np.argmin(fold_test_losses)

import csv
name = "BO_CV_2025_GIN_full_wo_sol"
csv1 = 'result/'+name+'.csv'
best_model = GINMultiGraphModelwoSol(
        input_dim_a=train_dataset[0][0].num_node_features,
        input_dim_c=train_dataset[0][1].num_node_features if train_dataset[0][1] is not None else 2,
        embedding_size=embedding_size,
        hl_size=hl_size,
        n_gnn_layers=n_gnn_layers,
        n_hl=n_hl,
        pooling=pooling,
        activation=activation,
        dropout=dropout
    ).to(device)
best_model.load_state_dict(torch.load(f"ckpt/model_fold{best_fold}.pth"))
best_model.eval()
test_dataset = MoleculesDataset(df_test)
test_loader = PyGDataLoader(test_dataset, batch_size=bs, shuffle=False)
test_loss = 0
with open(csv1, 'w', newline='') as file:
    fieldnames = [
    'epoch', 'predicted_x', 'predicted_y',
    'gt_x', 'gt_y', 'SMILES_flu', 'SMILES_acid', 'concentration'
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    with torch.no_grad():
        for data in test_loader:
            data1, data3, con, y = data
            data1, data3 = data1.to(device), data3.to(device)
            con, y = con.to(device), y.to(device)
            output = best_model(data1, data3, con)
            loss = mae(output, y)
            test_loss += loss.item()
            for i in range(output.shape[0]):
                data_for_csv = {
                'epoch': epoch + 1,
                'predicted_x': output[i].tolist()[0],
                'predicted_y': output[i].tolist()[1],
                'gt_x': y[i].squeeze().tolist()[0],
                'gt_y': y[i].squeeze().tolist()[1],
                'SMILES_flu': data1.smiles[i],
                'SMILES_acid': data3.smiles[i],
                'concentration': con[i].item()
                
                    }
                # Write the row to the CSV file
              
                writer.writerow(data_for_csv)
test_loss /= len(test_dataset)
print(f"Fold {best_fold} test loss: {test_loss}")



