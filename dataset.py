from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader
import numpy as np

class MoleculesDataset(Dataset):
    def __init__(self, dataset_a, dataset_b, dataset_c, con, target):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.dataset_c = dataset_c
        self.concentrations = con
        self.target = target

    def __len__(self):
        return len(self.dataset_a)

    def __getitem__(self, idx):
        data_a = self.dataset_a[idx]
        data_b = self.dataset_b[idx]
        data_c = self.dataset_c[idx]
        con = self.concentrations[idx]
        target = self.target[idx]
        return data_a, data_b, data_c, con, target


def atom_features(atom):
    # Example: Atomic number, Degree, Formal Charge, Hybridization
    return torch.tensor([atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), atom.GetHybridization().real, atom.IsInRing()])

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    atoms = mol.GetAtoms()
    node_features = torch.stack([atom_features(atom) for atom in atoms])

    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[start, end], [end, start]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=node_features, edge_index=edge_index, edge_attr=None)


#NOTE: Below are the two augmentation ways, given a SMILES, create a list of SMILES augmentations, one is based on Stereoisomers, the other is based on randomization of the SMILES string.
def generate_stereoisomers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Enumerate stereoisomers
    stereo_enumerator = AllChem.EnumerateStereoisomers
    opts = stereo_enumerator.EnumerationOptions(onlyUnassigned=True)
    stereoisomers = list(stereo_enumerator.EnumerateStereoisomers(mol, options=opts))
    # Convert to SMILES
    smiles_list = [Chem.MolToSmiles(isomer, isomericSmiles=True) for isomer in stereoisomers]
    return smiles_list

def generate_augmented_smiles(mol, num_augmentations=4):
    """
    Generate a specified number of randomized SMILES strings for a given RDKit molecule.
    Ensures the generated SMILES are different from the original.

    Parameters:
        mol (rdkit.Chem.Mol): RDKit molecule object.
        num_augmentations (int): Number of random SMILES to generate.

    Returns:
        list: List of randomized SMILES strings.
    """
    if mol is None:
        raise ValueError("Input molecule is None.")

    original_smiles = Chem.MolToSmiles(mol, canonical=True)
    augmented_smiles = set()

    while len(augmented_smiles) < num_augmentations:
        atom_indices = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_indices)
        randomized_mol = Chem.RenumberAtoms(mol, atom_indices)
        randomized_smiles = Chem.MolToSmiles(randomized_mol, canonical=False)

        # Ensure the new SMILES is not the same as the original or already generated ones
        if randomized_smiles != original_smiles:
            augmented_smiles.add(randomized_smiles)

    return list(augmented_smiles)



def combined_data_convert(df):
    flu_lewis_graphs = []
    targets = []
    con_value = []

    for n, row in df.iterrows():
        # Combine 'flu' and 'lewis' using the specified rule
        lewis_value = row['levis'] if pd.notna(row['levis']) else ''
        combined_smiles = f"{row['flu']}.{lewis_value}" if lewis_value else row['flu']
        print(combined_smiles)

        try:
            graph = smiles_to_graph(combined_smiles)
            graph.x = graph.x.float()
            graph.smiles = combined_smiles
            flu_lewis_graphs.append(graph)
        except Exception as e:
            print(f"Error processing SMILES: {combined_smiles}, Error: {e}")
            flu_lewis_graphs.append(None)  # Append None or a default value to maintain list length

        target = torch.tensor([row['x'], row['y']], dtype=torch.float)
        targets.append(target.squeeze())

        try:
            con_value.append(row['con'])
        except KeyError:
            con_value.append(None)  # Use None for missing 'con' values

    return flu_lewis_graphs, con_value, targets


def data_convert(df):
    flu_graphs = []
    acid_graphs = []
    sol_graphs = []
    targets = []
    con_value = []

    sol_rep = 0
    acid_rep = 0
    for n, row in df.iterrows():
        graph = smiles_to_graph(row['flu'])
        target = torch.tensor([row['x'], row['y']], dtype=torch.float)

        targets.append(target.squeeze())
        graph.x = graph.x.float()
        graph.smiles = row['flu']
        flu_graphs.append(graph)

        try:
            graph1 = smiles_to_graph(row['sol'])
            graph1.x = graph1.x.float()
            graph1.smiles = row['sol']
            sol_graphs.append(graph1)
            if n == 0:
                sol_rep=graph1
        except:
            sol_graphs.append(1)
            print("NO sol will appear in training!")

        try:
            if isinstance(row['levis'], float):
                row['levis'] = 'C'
            graph2 = smiles_to_graph(row['levis'])
            graph2.x = graph2.x.float()
            graph2.smiles = row['levis']
            acid_graphs.append(graph2)
            if not isinstance(row['levis'], float):
                    acid_rep=graph2
        except:
            acid_graphs.append(1)

        try:
            con_value.append(row['con'])  
        except:
            con_value.append(1)
            
    return flu_graphs, sol_graphs, acid_graphs, con_value, targets
        
        
def collate(samples):
    batched_data_a = DataLoader([s[0] for s in samples])
    batched_data_b = DataLoader([s[1] for s in samples])
    batched_data_c = DataLoader([s[2] for s in samples])
    batched_data_con = DataLoader([s[3] for s in samples])
    targets = DataLoader([s[4] for s in samples])
    return batched_data_a, batched_data_b, batched_data_c, batched_data_con, targets


class CombinedMoleculesDataset(Dataset):
    def __init__(self, dataset_a, con, target):
        self.dataset_a = dataset_a
        
        self.concentrations = con
        self.target = target

    def __len__(self):
        return len(self.dataset_a)

    def __getitem__(self, idx):
        data_a = self.dataset_a[idx]
        
        con = self.concentrations[idx]
        target = self.target[idx]
        return data_a, con, target
    
    
def data_convert_candidate(df, sol, acid, con):
    flu_graphs = []
    acid_graphs = []
    sol_graphs = []
    targets = []
    con_value = []

    sol_rep = 0
    acid_rep = 0
    for n, row in df.iterrows():
        graph = smiles_to_graph(row['flu'])
        target = torch.tensor([0,0], dtype=torch.float)

        targets.append(target.squeeze())
        graph.x = graph.x.float()
        graph.smiles = row['flu']
        flu_graphs.append(graph)

        if sol is not None:
            graph1 = smiles_to_graph(sol)
            graph1.x = graph1.x.float()
            graph1.smiles = sol
            sol_graphs.append(graph1)

        if acid is not None:
            graph2 = smiles_to_graph(acid)
            graph2.x = graph2.x.float()
            graph2.smiles = acid
            acid_graphs.append(graph2)

        if con is not None:
            con_value.append(con)  
            
    return flu_graphs, sol_graphs, acid_graphs, con_value, targets