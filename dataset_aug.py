import torch
import random
import numpy as np
from rdkit import Chem

from torch.utils.data import Dataset
from torch_geometric.data import Data
from auglichem.utils import ATOM_LIST, CHIRALITY_LIST

class MoleculesDataset(Dataset):
    def __init__(self, df, transform=None):
        self._x = df["x"].values.astype(np.float32)
        self._y = df["y"].values.astype(np.float32)
        self.con_values = df["con"].values.astype(np.float32)
        self.flu = df["flu"].tolist()
        self.sol = df["sol"].tolist() if "sol" in df.columns else None
        self.acid = df["levis"].tolist()
        self.transform = transform

    def _get_data_x(self, mol):
        # Set up data arrays
        type_idx, chirality_idx, atomic_number = [], [], []
        # Gather atom data
        for idx, atom in enumerate(mol.GetAtoms()):
            try:
                type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
                chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                atomic_number.append(atom.GetAtomicNum())
            except ValueError:  # Skip asterisk in motif
                pass
        # Concatenate atom type with chirality index
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)
        return x

    def _get_edge_index_and_attr(self, mol):
        """Convert molecule to graph edge indices."""
        adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
        edge_attr = torch.ones(
            (edge_index.shape[1], 1)
        )  # Default edge attributes (can be improved)
        return edge_index, edge_attr

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        atom_features = self._get_data_x(mol)
        edge_index, edge_attr = self._get_edge_index_and_attr(mol)

        return Data(
            x=atom_features, edge_index=edge_index, edge_attr=edge_attr, mol=mol, smiles=smiles
        )

    def mol_to_graph(self, mol, smiles):
        atom_features = self._get_data_x(mol)
        edge_index, edge_attr = self._get_edge_index_and_attr(mol)
        return Data(
            x=atom_features, edge_index=edge_index, edge_attr=edge_attr, mol=mol, smiles=smiles
        )

    def __len__(self):
        return len(self.flu)

    def __getitem__(self, index):
        mol_flu = self.smiles_to_graph(self.flu[index])
        if self.sol is not None:
            mol_sol = self.smiles_to_graph(self.sol[index])
    
        mol_acid = self.smiles_to_graph(self.acid[index])
        con_value = torch.tensor(self.con_values[index], dtype=torch.float)
        target = torch.tensor([self._x[index], self._y[index]], dtype=torch.float)

        if self.transform:
            mol_flu = self.transform(mol_flu)
            selected_mol = random.choice(mol_flu)
            mol_flu = self.mol_to_graph(selected_mol, self.flu[index])
            # mol_y = self.transform(mol_y)
        if self.sol is not None:
            return mol_flu, mol_sol, mol_acid, con_value, target
        else:   
            return mol_flu, mol_acid, con_value, target