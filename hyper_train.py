import optuna
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
seed = 1

# Objective function for hyperparameter tuning with Optuna
def objective(trial):
    # Define hyperparameter search space
    embedding_size = trial.suggest_int("embedding_size", 10, 300)
    hl_size = trial.suggest_int("hl_size", 10, 400)
    n_gnn_layers = trial.suggest_int("n_gnn_layers", 2, 4)
    n_hl = trial.suggest_int("n_hl", 1, 3)
    pooling = trial.suggest_categorical("pooling", ["add", "mean", "max"])
    activation = trial.suggest_categorical("activation", ["ReLU", "leaky_ReLU"])
    learning_rate = trial.suggest_float("lr", 0.00005, 0.001, log=True)
    bs = trial.suggest_int("bs", 1, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load and preprocess dataset
    df = pd.read_csv("new_data/xy full.csv", na_values=["None"], header=None, skiprows=1)
    df.columns = ["x", "y", "con", "flu", "levis"]
    df["levis"] = df["levis"].fillna("C").astype(str)

    # Split into training (90%) and independent test set (10%) NEVER USED HERE
    df_train, _ = train_test_split(df, test_size=0.1, random_state=seed) 

    # Perform 5-fold cross-validation on training set
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    val_losses = []

    for train_idx, val_idx in kf.split(df_train):
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

        best_val_loss = float("inf")
        epochs_no_improve = 0

        # Training loop with early stopping
        for epoch in range(4000):  # Reduced epochs for efficiency
            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                data1, data3, con, y = data
                data1, data3, con, y = data1.to(device), data3.to(device), con.to(device), y.to(device)
                output = model(data1, data3, con)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            # Evaluate validation loss
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data1, data3, con, y = data
                    data1, data3, con, y = data1.to(device), data3.to(device), con.to(device), y.to(device)
                    output = model(data1, data3, con)
                    loss = mae(output, y)
                    val_loss += loss.item()
            val_loss /= len(val_dataset)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= 50:
                break

        val_losses.append(best_val_loss)

    # Return average validation loss across folds
    return np.mean(val_losses)

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Evaluate best hyperparameters on independent test set
print("Best hyperparameters:", study.best_params)
print("Best CV validation value:", study.best_value)
