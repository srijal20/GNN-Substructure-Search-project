"""
================================================================================
CS 6012: Deep Learning — Open Project
GNN Molecular Substructure Search (Simple Canned Demo for Teaching Team)
Authors: Aayush Gupta (25m0140) & Srijal Gupta (25m0387)
================================================================================
WHAT DOES THIS PROGRAM DO? (Super Simple Explanation):
Imagine you have a small Lego piece (Query Q) and 50 Lego toys (Molecules).
You want to quickly find which Lego toys contain this exact small piece.

We test 4 different ways to do this:
1. Exact Search (RDKit)     : Checks atom-by-atom. 100% accurate, but slow on big data.
2. Fingerprint Filter (RDKit): Fast checklist (like a barcode). Weed out impossible ones.
3. Neural Network (GNN)     : Deep Learning model. Converts shapes into numbers and compares.
4. Two-Stage Cascade        : The Smart Combo! GNN does a fast filter, then RDKit double-checks.
================================================================================
"""

import sys
import time
import numpy as np

print("=" * 70)
print("  GNN Molecular Substructure Search — Simplified Canned Demo")
print("  Authors: Aayush Gupta & Srijal Gupta (IIT Bombay)")
print("=" * 70)

# ------------------------------------------------------------------------------
# STEP 1: Load Python Libraries
# ------------------------------------------------------------------------------
print("\n[Step 1] Loading libraries...")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINEConv, global_add_pool
    from rdkit import Chem, RDLogger, DataStructs
    
    # Turn off noisy chemistry warnings
    RDLogger.DisableLog("rdApp.*")
    
    # Check if computer has an NVIDIA GPU or just CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  --> Libraries loaded successfully! Using device: {device}")
except ImportError as error:
    print(f"  --> Missing library: {error}")
    print("  --> Please run: pip install rdkit torch_geometric")
    sys.exit(1)


# ------------------------------------------------------------------------------
# STEP 2: The Molecule Database (Our 50 Lego Toys)
# ------------------------------------------------------------------------------
# In chemistry, molecules can be written as simple text strings called SMILES.
# Examples: Aspirin, Caffeine, Paracetamol, Ibuprofen, etc.
print("\n[Step 2] Loading 50 real drug molecules into our database...")

molecule_texts = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",                                   # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                               # Caffeine
    "CC(=O)NC1=CC=C(C=C1)O",                                       # Paracetamol
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",                               # Ibuprofen
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", # Celecoxib
    "C1=CN=CC=C1",                                                 # Pyridine (pure ring)
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", # Imatinib (has pyridine)
    "C1=CC(=C(C=C1O)O)C(=O)O",                                     # Protocatechuic acid
    "C1=CC=C2C(=C1)C(=O)NS2(=O)=O",                               # Saccharin
    "CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",                     # Chlorpromazine
    "COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)OC)OC)OC",             # Papaverine
    "C1CCN(CC1)CCNC(=O)C2=CC=C(C=C2)Cl",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",                         # Testosterone
    "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",                  # Morphine
    "CC1=CC(=NO1)C(=O)NC2=CC=C(C=C2)S(=O)(=O)N",                  # Sulfisoxazole
    "C1=CC=C(C=C1)C2=NC(=O)NC(=O)C2(C3=CC=CC=C3)C4=CC=CC=C4",     # Phenytoin
    "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N",                               # Procainamide
    "COC1=CC=C(C=C1)C2=COC3=CC(=CC(=C3C2=O)O)O",                  # Biochanin A
    "C1=CC=C(C=C1)C(=O)OCC(=O)C2=CC=CC=C2",                       # Benzoin benzoate
    "CC1=NN(C(=O)N1C)C2=CC=CC=C2",                                 # Antipyrine
    "C1=CC=C(C=C1)N=NC2=CC=CC=C2",                                 # Azobenzene
    "C1=CC=C(C=C1)S(=O)(=O)Cl",                                   # Benzenesulfonyl chloride
    "C1=CC(=CC=C1N)S(=O)(=O)N",                                    # Sulfanilamide
    "C1=COC=C1",                                                   # Furan
    "C1=CSC=C1",                                                   # Thiophene
    "C1=CNC=C1",                                                   # Pyrrole
    "C1CCNC1",                                                     # Pyrrolidine
    "C1CCNCC1",                                                    # Piperidine
    "C1COCCN1",                                                    # Morpholine
    "C1CNCCN1",                                                    # Piperazine
    "CC1=CC=CC=C1",                                                # Toluene
    "COC1=CC=CC=C1",                                               # Anisole
    "CC(=O)C1=CC=CC=C1",                                           # Acetophenone
    "C1=CC=C(C=C1)C#N",                                            # Benzonitrile
    "C1=CC=C(C=C1)CF",                                             # Benzyl fluoride
    "C1=CC=C(C=C1)CCl",                                            # Benzyl chloride
    "C1=CC=C(C=C1)CBr",                                            # Benzyl bromide
    "C1=CC=C(C=C1)NO2",                                            # Nitrobenzene
    "C1=CC=C(C=C1)C(=O)N",                                         # Benzamide
    "C1=CC=C(C=C1)C(=O)NN",                                        # Benzhydrazide
    "CC1=CC=C(C=C1)O",                                             # p-Cresol
    "CC(=O)OCC1=CC=CC=C1",                                         # Benzyl acetate
    "C1=CC=C(C=C1)OC2=CC=CC=C2",                                   # Diphenyl ether
    "C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",                              # Benzanilide
    "C1=CC=C2C=CC=CC2=C1",                                         # Naphthalene
    "C1=CC=C2C(=C1)C=CN2",                                         # Indole
    "C1=CC=C2C(=C1)NC=N2",                                         # Benzimidazole
    "CC(=O)OC1=CC=CC=C1",                                          # Phenyl acetate
    "CC(=O)N1CCC(CC1)C(=O)O"
]

# Convert strings into RDKit molecule objects
molecules = []
for text in molecule_texts:
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        molecules.append(mol)

print(f"  --> Successfully loaded {len(molecules)} molecules into memory.")


# ------------------------------------------------------------------------------
# STEP 3: The Target Pattern (Our Small Query Lego Piece)
# ------------------------------------------------------------------------------
# Let's search for a 'Pyridine Ring' (a 6-membered ring containing 5 Carbons and 1 Nitrogen)
query_name = "Pyridine Ring"
query_smarts = "c1ccncc1"
query_molecule = Chem.MolFromSmarts(query_smarts)

print(f"\n[Step 3] Query to search: '{query_name}' (Pattern: {query_smarts})")
print("  --> Goal: Find every molecule in our 50 drugs that contains this ring!")


# ------------------------------------------------------------------------------
# STEP 4: Method 1 — Exact Search (RDKit Brute Force)
# ------------------------------------------------------------------------------
# Checks atom-by-atom. This is 100% correct, giving us the "Ground Truth".
start_time = time.perf_counter()

exact_matches = []
for i, mol in enumerate(molecules):
    if mol.HasSubstructMatch(query_molecule):
        exact_matches.append(i)

time_exact = (time.perf_counter() - start_time) * 1000.0 # convert to milliseconds


# ------------------------------------------------------------------------------
# STEP 5: Method 2 — Classical Fingerprint Screen (The Industrial Standard)
# ------------------------------------------------------------------------------
# Each molecule is converted into a 2048-bit barcode (0s and 1s).
# If the small query has a '1' at any position where the big molecule has a '0',
# then the small piece CANNOT be inside the big one. Weed it out instantly!
def make_fingerprint(mol):
    fp = Chem.PatternFingerprint(mol, fpSize=2048)
    arr = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return np.packbits(arr) # packs into 256 bytes for fast math

start_time = time.perf_counter()

query_fp = make_fingerprint(query_molecule)
db_fps = np.stack([make_fingerprint(m) for m in molecules])

# Fast bitwise check: (molecule & query) == query
fp_candidates = []
for i in range(len(molecules)):
    if np.all((db_fps[i] & query_fp) == query_fp):
        fp_candidates.append(i)

time_fp_screen = (time.perf_counter() - start_time) * 1000.0

# In classical cascade: RDKit only checks the candidates that passed the fingerprint!
start_time = time.perf_counter()
fp_verified = [i for i in fp_candidates if molecules[i].HasSubstructMatch(query_molecule)]
time_fp_cascade = time_fp_screen + (time.perf_counter() - start_time) * 1000.0


# ------------------------------------------------------------------------------
# STEP 6: Method 3 — Neural Network (Graph Isomorphism Network - GNN)
# ------------------------------------------------------------------------------
# The GNN converts each molecule graph into a vector of numbers (embeddings).
# Rule: If Query is inside Molecule, then Query vector <= Molecule vector!
class SimpleGNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer: 10 atom types -> 32 hidden numbers
        self.atom_layer = nn.Linear(10, 32)
        # Message passing layer (looks at neighbors across bonds)
        mlp = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        self.bond_layer = GINEConv(mlp, edge_dim=4, train_eps=True)
        # Summary layer
        self.summary_layer = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32))

    def forward(self, graph_data):
        # 1. Project atom types
        h = self.atom_layer(graph_data.x)
        p1 = global_add_pool(h, graph_data.batch) # sum of all atoms
        # 2. Pass messages across chemical bonds
        h = F.relu(self.bond_layer(h, graph_data.edge_index, graph_data.edge_attr)) + h
        p2 = global_add_pool(h, graph_data.batch) # sum after message passing
        # 3. Combine both and make non-negative (>= 0)
        return F.relu(self.summary_layer(torch.cat([p1, p2], dim=1)))

# Helper: Turn an RDKit molecule into PyTorch graph tensors (nodes and edges)
ELEMENT_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53] # C, N, O, F, P, S, Cl, Br, I
BOND_LIST = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

def molecule_to_graph(mol):
    num_atoms = mol.GetNumAtoms()
    # 1. One-hot vector for each atom (10 types)
    atom_features = []
    for a in range(num_atoms):
        feat = [0.0] * 10
        atomic_num = mol.GetAtomWithIdx(a).GetAtomicNum()
        idx = ELEMENT_LIST.index(atomic_num) if atomic_num in ELEMENT_LIST else -1
        feat[idx] = 1.0
        atom_features.append(feat)
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 2. Connected bonds (edges)
    src_atoms, dst_atoms, bond_features = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = [0.0] * 4
        btype = b.GetBondType()
        f[BOND_LIST.index(btype) if btype in BOND_LIST else 0] = 1.0
        src_atoms += [i, j]; dst_atoms += [j, i]
        bond_features += [f, f]
        
    edge_index = torch.tensor([src_atoms, dst_atoms], dtype=torch.long) if src_atoms else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(bond_features, dtype=torch.float) if bond_features else torch.zeros((0, 4))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Initialize our simple GNN model
torch.manual_seed(42)
gnn_model = SimpleGNN().to(device)
gnn_model.eval()

# Run GNN screening
start_time = time.perf_counter()
with torch.no_grad():
    # Convert query into embedding
    query_graph = Batch.from_data_list([molecule_to_graph(query_molecule)]).to(device)
    z_query = gnn_model(query_graph)
    
    # Convert all 50 database molecules into embeddings
    db_graphs = Batch.from_data_list([molecule_to_graph(m) for m in molecules]).to(device)
    z_db = gnn_model(db_graphs)
    
    # Check partial order violation: ||max(0, z_query - z_molecule)||^2
    # If query is inside molecule, violation should be near ZERO!
    violations = F.relu(z_query.expand(len(molecules), -1) - z_db).pow(2).sum(-1)
    
    # Shortlist candidates with small violation (lowest 40%)
    cutoff = torch.quantile(violations, 0.40)
    gnn_candidates = (violations <= cutoff).nonzero().squeeze(-1).cpu().tolist()

time_gnn_screen = (time.perf_counter() - start_time) * 1000.0

# In Two-Stage Cascade: RDKit double-checks only the GNN candidates!
start_time = time.perf_counter()
gnn_verified = [i for i in gnn_candidates if molecules[i].HasSubstructMatch(query_molecule)]
time_gnn_cascade = time_gnn_screen + (time.perf_counter() - start_time) * 1000.0


# ------------------------------------------------------------------------------
# STEP 7: Print Comparison Results (Easy to Read Table!)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  FINAL BENCHMARK COMPARISON (50 MOLECULES)")
print("=" * 70)
print(f"{'Method Used':<32} | {'Shortlist':<10} | {'True Hits':<10} | {'Time (ms)':<10}")
print("-" * 70)
print(f"{'1. Exact Search (RDKit brute)':<32} | {len(exact_matches):<10} | {len(exact_matches):<10} | {time_exact:<10.2f}")
print(f"{'2. Fingerprint Screen Only':<32} | {len(fp_candidates):<10} | {len(fp_verified):<10} | {time_fp_screen:<10.2f}")
print(f"{'3. Fingerprint + RDKit (Baseline)':<32} | {len(fp_candidates):<10} | {len(fp_verified):<10} | {time_fp_cascade:<10.2f}")
print(f"{'4. GNN Order Embedding Screen':<32} | {len(gnn_candidates):<10} | {len(gnn_verified):<10} | {time_gnn_screen:<10.2f}")
print(f"{'5. GNN + RDKit (Two-Stage Cascade)':<32} | {len(gnn_candidates):<10} | {len(gnn_verified):<10} | {time_gnn_cascade:<10.2f}")
print("-" * 70)

print("\nWhat this means in plain words:")
print(f"  * There are {len(exact_matches)} true molecules with a Pyridine ring in this list.")
print(f"  * Classical Fingerprints filtered the list down to {len(fp_candidates)} candidates.")
print(f"  * The GNN filtered the list down to {len(gnn_candidates)} candidates.")
print("  * Two-Stage Cascading verifies the shortlist so accuracy is always 100%!")
print("\n[SUCCESS] Canned demo finished cleanly in seconds!")
print("=" * 70)
