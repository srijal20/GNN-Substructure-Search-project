# %% [markdown]
# # GNN Molecular Substructure Search: Accuracy vs. Speed Trade-offs Against Classical Fingerprint Screening
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srijal20/GNN-Substructure-Search-project/blob/main/GNN_Substructure_Search.ipynb)
#
# **Course:** CS 6012: Deep Learning — Open Project (Submission 2)  
# **Authors:** Aayush Gupta (25m0140) & Srijal Gupta (25m0387), IIT Bombay  
#
# ---
#
# ### 📖 What is Molecular Substructure Search? (Beginner-Friendly Overview)
#
# In modern drug discovery and cheminformatics, medicinal chemists frequently need to query massive chemical databases (often containing tens of millions to billions of molecules) to answer a fundamental question:
# > **"Which molecules in our chemical library contain this specific functional group, core scaffold, or pharmacophore pattern $Q$?"**
#
# #### The Computational Challenge: Subgraph Isomorphism
# - A molecule can be naturally represented as an undirected graph $G = (V, E)$, where vertices $V$ represent **atoms** and edges $E$ represent **chemical bonds**.
# - Determining whether a query fragment $Q$ is structurally embedded inside a target molecule $T$ corresponds to the **Subgraph Isomorphism Problem**.
# - Subgraph isomorphism is mathematically proven to be **NP-complete**. Combinatorial graph matching algorithms (such as Ullmann, VF2, or Cordella's algorithm implemented in RDKit) exhibit exponential worst-case computational complexity $\mathcal{O}(|V_T|^{|V_Q|})$. Running exact matching on millions of molecules is prohibitively slow.
#
# #### The Industrial Standard: Bitwise Screening
# To bypass this combinatorial bottleneck, production cheminformatics suites use a **Two-Stage Filtering Architecture**:
# 1. **Screening Stage (Fast):** Compute a bit-vector summary (fingerprint) for every molecule. Filter the database using fast bitwise operations.
# 2. **Verification Stage (Accurate):** Execute exact combinatorial graph matching **only** on the small candidate subset that survived the screening filter.
#
# RDKit's **Pattern Fingerprint** (`PatternFingerprint`) is the industry benchmark for screening because it guarantees **zero false negatives (100% recall)**: if $Q$ is a true subgraph of $T$, every bit set in the fingerprint of $Q$ must also be set in the fingerprint of $T$.
#
# #### The Research Question of this Project
# > **"Can a learned Graph Neural Network (GNN) outperform or augment classical cheminformatics fingerprint screening in latency and candidate selectivity, while maintaining near-perfect recall?"**
#
# ---
#
# ### 🔬 Evaluated Approaches in this Notebook
# We implement, train, and empirically benchmark six distinct retrieval pipelines on the **ZINC250k** molecular database:
# 1. **RDKit Exact (Brute Force):** Pure combinatorial matching on all molecules (`mol.HasSubstructMatch`). Serves as our gold-standard ground truth.
# 2. **Pattern Fingerprint Screen Only:** Fast bitwise containment check using 2048-bit RDKit Pattern Fingerprints.
# 3. **Classical Two-Stage Baseline (Pattern FP + RDKit):** Bitwise screening followed by exact RDKit verification.
# 4. **Model A (Siamese GIN + MLP Classifier):** A Graph Isomorphism Network that embeds both $Q$ and $T$, then feeds their concatenated representations into an MLP classifier trained with Binary Cross-Entropy (BCE).
# 5. **Model B (NeuroMatch Order Embedding):** A geometric GNN that maps molecules into a non-negative cone $\mathbb{R}_{\ge 0}^d$, enforcing a strict partial order: $Q \subseteq T \iff \mathbf{z}_Q \le \mathbf{z}_T$ element-wise.
# 6. **Two-Stage Neural Cascade (GNN B Shortlist + RDKit Verification):** A practical hybrid pipeline where the GNN order embedding acts as a rapid candidate filter (calibrated at $\ge 98\%$ recall), and RDKit verifies only the shortlisted candidates.
#
# ---
#
# ### 🚀 Execution Guide
# - **Hardware Requirement:** Run on a GPU runtime (**Runtime > Change runtime type > T4 GPU**).
# - **Fast Smoke Test vs. Full Paper Benchmark:**
#   - Set `QUICK = True` in Section 1 to run a quick 5-minute sanity check on 5,000 DB molecules.
#   - Set `QUICK = False` to reproduce the full 50,000 molecule paper benchmark (~25–35 minutes on GPU).

# %%
# ==============================================================================
# CELL 1: Automated Package Installation
# ==============================================================================
# In this cell, we ensure all necessary domain-specific libraries are installed.
# When running on Google Colab, PyTorch is pre-installed, but specialized chemistry
# and graph learning toolkits must be fetched via pip:
#   - rdkit: The industry standard open-source library for cheminformatics.
#   - torch_geometric (PyG): The premier graph deep learning framework for PyTorch.
# ==============================================================================

import importlib.util
import subprocess
import sys

# Define required third-party libraries: (pip package name, importable module name)
REQUIRED_PACKAGES = [
    ("rdkit", "rdkit"),                         # Molecular representations, SMILES parsing, fingerprints
    ("torch_geometric", "torch_geometric")       # GNN layers (GINEConv), graph pooling, mini-batching
]

for package_name, module_name in REQUIRED_PACKAGES:
    # Check if the module is already importable in the current Python environment
    if importlib.util.find_spec(module_name) is None:
        print(f"--> Installing missing package: '{package_name}'...")
        # Run pip install quietly (-q) using the currently executing python interpreter
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])
        print(f"--> Successfully installed '{package_name}'.")
    else:
        print(f"--> Package '{package_name}' is already installed and ready.")


# %%
# ==============================================================================
# CELL 2: Module Imports & Hardware Accelerator Setup
# ==============================================================================
# We import:
# 1. Standard Python utilities (os, time, random, math, json).
# 2. Scientific computing libraries (NumPy, Pandas, Matplotlib).
# 3. PyTorch deep learning modules (torch, nn, F).
# 4. PyTorch Geometric graph primitives:
#    - Data: Represents a single molecular graph with node and edge attributes.
#    - Batch: Batches multiple independent graphs into a single block-diagonal super-graph.
#    - GINEConv: Graph Isomorphism Network convolution supporting edge features.
#    - global_add_pool: Graph readout layer summing node features for each molecule.
# 5. RDKit tools for SMILES parsing and fingerprint computation.
# ==============================================================================

import os
import time
import random
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch core deep learning modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric (PyG) graph data structures and GNN layers
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_add_pool

# RDKit cheminformatics modules
from rdkit import Chem, RDLogger, DataStructs

# Suppress minor RDKit chemistry warnings (e.g., unusual synthetic valences)
# to keep notebook output clean and professional
RDLogger.DisableLog("rdApp.*")

# Automatically select NVIDIA CUDA GPU if available; otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Compute Device : {DEVICE}")
print(f"PyTorch Version       : {torch.__version__}")
if DEVICE.type == "cuda":
    print(f"NVIDIA GPU Model      : {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")


# %% [markdown]
# ## 1. Experiment Configuration & Hyperparameters
#
# All dataset sizes, network dimensions, and training hyperparameters are centralized in the `CFG` dictionary below.
#
# ### Why two modes (`QUICK = True` vs. `QUICK = False`)?
# - **`QUICK = True` (Sanity Check Mode):** Trains for 4 epochs on 4,000 molecules, searching a 5,000-molecule database. Takes ~5 minutes on a standard T4 GPU. Use this to verify that the entire pipeline runs without errors.
# - **`QUICK = False` (Full Scientific Benchmark):** Trains for 20 epochs on 40,000 molecules, searching a 50,000-molecule database with 300 test queries. Matches the exact empirical results reported in our CS 6012 final report.
#
# ### Hyperparameters Overview:
# - `QUERY_MIN_ATOMS` (4) & `QUERY_MAX_ATOMS` (12): Realistic size range for drug discovery functional groups (e.g., benzene rings, carboxyls, sulfonamides).
# - `HIDDEN` (64) & `EMB_DIM` (64): Latent vector dimension for node features and graph-level embeddings ($\mathbf{z} \in \mathbb{R}^{64}$).
# - `N_LAYERS` (4): Number of GINE message-passing layers. Each layer expands the receptive field by 1 hop, allowing the GNN to capture topological neighborhoods up to 4 bonds away.
# - `LR` (1e-3): Learning rate for the Adam optimizer.
# - `MARGIN` (1.0): Hinge margin $\gamma$ for the NeuroMatch order-embedding loss.
# - `FP_BITS` (2048): Bit-width of the classical RDKit Pattern Fingerprint.

# %%
# ==============================================================================
# CELL 4: Configuration Settings & Random Seed Initialization
# ==============================================================================

# Set QUICK to True for a quick 5-minute sanity test, or False for the full benchmark
QUICK = False
SEED = 0

if QUICK:
    # Lightweight configuration for rapid validation
    CFG = dict(
        N_TRAIN_MOLS=4000,          # Molecules used to construct training pairs
        N_VAL_MOLS=500,             # Molecules used for validation & threshold tuning
        N_DB=5000,                  # Search database size (held-out molecules)
        N_QSRC=200,                 # Query source molecules (disjoint from train and DB)
        N_TRAIN_QUERIES=4000,       # Number of query fragments sampled for training
        N_VAL_QUERIES=500,          # Number of queries for validation
        N_TEST_QUERIES=100,         # Number of test queries for evaluation
        EPOCHS=4,                   # Number of training epochs
        BATCH=256                   # Mini-batch size
    )
else:
    # Full experimental configuration reported in the paper
    CFG = dict(
        N_TRAIN_MOLS=40000,         # 40,000 training molecules
        N_VAL_MOLS=4000,            # 4,000 validation molecules
        N_DB=50000,                 # 50,000 search database molecules
        N_QSRC=2000,                # 2,000 held-out query source molecules
        N_TRAIN_QUERIES=40000,      # 40,000 training queries
        N_VAL_QUERIES=3000,         # 3,000 validation queries
        N_TEST_QUERIES=300,         # 300 test queries
        EPOCHS=20,                  # 20 full training epochs
        BATCH=256                   # Mini-batch size
    )

# Model architecture and training hyperparameters
CFG.update(
    QUERY_MIN_ATOMS=4,              # Minimum atom count for a sampled query fragment
    QUERY_MAX_ATOMS=12,             # Maximum atom count for a sampled query fragment
    HIDDEN=64,                      # Hidden channel dimension inside GNN layers
    EMB_DIM=64,                     # Final molecular graph embedding dimension (z in R^64)
    N_LAYERS=4,                     # Number of message passing layers (receptive field = 4 hops)
    LR=1e-3,                        # Adam optimizer initial learning rate
    MARGIN=1.0,                     # Margin gamma for order-embedding loss
    FP_BITS=2048                    # RDKit Pattern Fingerprint bit length
)

print("Active Configuration Parameters:")
for param, val in CFG.items():
    print(f"  {param:20s}: {val}")

# Seed all random number generators for strict reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Create an output directory for saving plots, CSVs, and model checkpoints
os.makedirs("results", exist_ok=True)
print("\n[OK] Results directory verified at './results'")


# %% [markdown]
# ## 2. Load Molecules (ZINC250k) and Disjoint Data Splitting
#
# We use **ZINC250k**, a curated dataset of approximately 250,000 commercially available drug-like compounds frequently used as a standard benchmark in computational chemistry and molecular generative modeling.
#
# ### The Critical Principle: Strictly Disjoint Splits
# To prevent **data leakage** and ensure that our evaluation strictly tests generalisation to unseen chemical space, we partition the dataset into four mutually disjoint subsets:
# 1. **`train_mols`:** Used to generate training query-target pairs for the GNN models.
# 2. **`val_mols`:** Used to monitor training loss, compute validation ROC-AUC, and tune the decision thresholds.
# 3. **`db_mols`:** The held-out search database (up to 50,000 molecules). **The GNN never sees these molecules during training!**
# 4. **`qsrc_mols`:** An independent pool of held-out molecules used solely to extract test queries.

# %%
# ==============================================================================
# CELL 6: Downloading ZINC250k & Constructing Disjoint Splits
# ==============================================================================
# 1. Download the ZINC250k dataset if not already present on disk.
# 2. Parse SMILES strings into RDKit Mol objects.
# 3. Partition the molecules into disjoint train, val, search DB, and query source pools.
# ==============================================================================

# URL of the standard clean 250k ZINC dataset from the Aspuru-Guzik group repository
ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)

# Download CSV file if not already cached locally
if not os.path.exists("zinc250k.csv"):
    import urllib.request
    print("Downloading ZINC250k dataset (~12 MB)...")
    urllib.request.urlretrieve(ZINC_URL, "zinc250k.csv")
    print("Download complete.")

# Read the SMILES (Simplified Molecular Input Line Entry System) strings
# Each line in the CSV contains a compact textual string representing a molecule
raw_df = pd.read_csv("zinc250k.csv")
smiles_all = [s.strip() for s in raw_df["smiles"]]
print(f"Total raw SMILES strings loaded: {len(smiles_all):,}")

# Shuffle the list of SMILES using a fixed random seed for reproducibility
rng = random.Random(SEED)
rng.shuffle(smiles_all)

# Determine total valid molecules needed across all splits
total_needed = CFG["N_TRAIN_MOLS"] + CFG["N_VAL_MOLS"] + CFG["N_DB"] + CFG["N_QSRC"]
mols_all = []

# Parse SMILES strings into RDKit Mol objects (atoms, bonds, connectivity)
for s in smiles_all:
    m = Chem.MolFromSmiles(s)
    if m is not None:                       # Filter out any synthetically malformed SMILES
        mols_all.append(m)
    if len(mols_all) == total_needed:
        break

# Slice into 4 strictly disjoint subsets
a = CFG["N_TRAIN_MOLS"]
b = a + CFG["N_VAL_MOLS"]
c = b + CFG["N_DB"]

train_mols = mols_all[:a]                   # Training set (for GNN pair learning)
val_mols   = mols_all[a:b]                  # Validation set (for threshold calibration)
db_mols    = mols_all[b:c]                  # Search database (the held-out chemical library)
qsrc_mols  = mols_all[c:]                   # Query sources (independent test query generator)

print(f"Data Split Summary:")
print(f"  - Training Molecules       : {len(train_mols):,}")
print(f"  - Validation Molecules     : {len(val_mols):,}")
print(f"  - Search Database (DB)     : {len(db_mols):,}")
print(f"  - Query Source Pool        : {len(qsrc_mols):,}")
print(f"  - Mean Atoms per DB Mol    : {np.mean([m.GetNumAtoms() for m in db_mols]):.1f} atoms")


# %% [markdown]
# ## 3. Molecular Graph Representation & Query Generation
#
# To feed molecules into Graph Neural Networks, we must translate chemical structures into graph tensors.
#
# ### Graph Formulation
# - **Nodes ($V$):** Heavy atoms (Carbon, Nitrogen, Oxygen, etc.). Hydrogen atoms are treated as *implicit* to keep graphs compact.
# - **Edges ($E$):** Chemical bonds connecting atoms. In PyG, undirected bonds are represented as pairs of directed edges $(u 	o v)$ and $(v 	o u)$.
#
# ### Feature Encoding:
# 1. **Node Features (`x`):** One-hot vector of length 10 indicating atomic number:
#    - Elements: $	ext{C (6), N (7), O (8), F (9), P (15), S (16), Cl (17), Br (35), I (53)}$ + 1 fallback slot for other elements.
# 2. **Edge Features (`edge_attr`):** One-hot vector of length 4 indicating chemical bond order:
#    - Bond Types: **Single, Double, Triple, Aromatic**.
#
# ### Query Generation:
# - **Sampled Connected Subgraphs:** We randomly select a seed atom and perform a breadth-first search (BFS) walk to extract a connected subgraph of $k \in [4, 12]$ atoms.
# - **Hand-Curated Functional Groups:** In addition to sampled subgraphs, we curate 20 standard medicinal functional groups (benzene, pyridine, ester, amide, sulfonamide, etc.) representing realistic queries issued by drug hunters.

# %%
# ==============================================================================
# CELL 8: Graph Featurization, Subgraph Extraction & Query Generation
# ==============================================================================
# Here we define:
# 1. atom_feat: Encodes an atom's element type as a one-hot vector (length 10).
# 2. bond_feat: Encodes a bond's type as a one-hot vector (length 4).
# 3. mol_to_graph: Converts an RDKit Mol into a PyTorch Geometric Data object.
# 4. sample_connected_atoms: BFS expansion to extract connected atom subgraphs.
# 5. sample_queries: Generates query objects with SMARTS and PyG graph representations.
# 6. fg_queries: Realistic hand-picked functional group queries (medicinal chemistry).
# ==============================================================================

# Atomic numbers common in drug-like organic molecules (C, N, O, F, P, S, Cl, Br, I)
ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53]

# The four standard chemical bond types
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC
]

def atom_feat(atom):
    """Converts an RDKit Atom into a 10-dimensional one-hot feature vector."""
    v = [0.0] * (len(ELEMENTS) + 1)
    z = atom.GetAtomicNum()
    # Map to index if in common list; otherwise place in final 'other' slot
    idx = ELEMENTS.index(z) if z in ELEMENTS else -1
    v[idx] = 1.0
    return v

def bond_feat(bond):
    """Converts an RDKit Bond into a 4-dimensional one-hot feature vector."""
    v = [0.0] * len(BOND_TYPES)
    bt = bond.GetBondType()
    idx = BOND_TYPES.index(bt) if bt in BOND_TYPES else 0
    v[idx] = 1.0
    return v

def mol_to_graph(mol, atom_ids=None):
    """
    Constructs a PyG Data object from an RDKit molecule or a selected subset of atoms.
    
    Parameters:
        mol: RDKit Mol object
        atom_ids: Optional list of integer atom indices (for subgraphs)
    Returns:
        Data(x, edge_index, edge_attr): PyTorch Geometric graph tensor
    """
    if atom_ids is None:
        atom_ids = list(range(mol.GetNumAtoms()))
        
    # Local index mapping: original atom idx -> 0..k-1
    pos = {a: i for i, a in enumerate(atom_ids)}
    
    # Node features: shape [num_nodes, 10]
    x = torch.tensor([atom_feat(mol.GetAtomWithIdx(a)) for a in atom_ids], dtype=torch.float)
    
    src, dst, ea = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Include bond only if BOTH constituent atoms are in the selected subgraph
        if i in pos and j in pos:
            f = bond_feat(bond)
            # PyG graphs are directed: add both forward and reverse edges
            src += [pos[i], pos[j]]
            dst += [pos[j], pos[i]]
            ea  += [f, f]
            
    # edge_index: shape [2, 2*num_bonds], edge_attr: shape [2*num_bonds, 4]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_attr  = torch.tensor(ea, dtype=torch.float) if ea else torch.zeros((0, len(BOND_TYPES)))
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def sample_connected_atoms(mol, k, rng):
    """
    Extracts a connected set of k atoms from a molecule using Breadth-First expansion.
    Connectivity is essential: disconnected atoms do not form valid chemical substructures.
    """
    # Pick a random starting seed atom
    sel = {rng.randrange(mol.GetNumAtoms())}
    while len(sel) < k:
        # Find all neighboring atoms not yet selected
        frontier = [
            n.GetIdx() 
            for a in sel 
            for n in mol.GetAtomWithIdx(a).GetNeighbors()
            if n.GetIdx() not in sel
        ]
        if not frontier:
            break                   # Molecule has fewer than k connected atoms
        sel.add(rng.choice(frontier))
    return sorted(sel)

def make_query(mol, atom_ids):
    """
    Constructs a query dictionary containing:
    - smarts: SMARTS string pattern
    - qmol: RDKit Mol query object
    - graph: PyG Data graph representation
    """
    flat = Chem.Mol(mol)
    # Remove stereochemistry flags (e.g., [C@@H]) so queries match 2D topological connectivity
    Chem.RemoveStereochemistry(flat)
    smarts = Chem.MolFragmentToSmarts(flat, atomsToUse=atom_ids)
    return dict(
        smarts=smarts,
        qmol=Chem.MolFromSmarts(smarts),
        graph=mol_to_graph(mol, atom_ids)
    )

def sample_queries(mols, n, rng):
    """Generates n connected subgraph queries sampled from the provided molecules."""
    out = []
    while len(out) < n:
        m = rng.choice(mols)
        k = rng.randint(CFG["QUERY_MIN_ATOMS"], min(CFG["QUERY_MAX_ATOMS"], m.GetNumAtoms()))
        q = make_query(m, sample_connected_atoms(m, k, rng))
        q["source"] = m             # Keep reference to the origin molecule
        out.append(q)
    return out

# Curated dictionary of common medicinal chemistry functional groups
FG_SMILES = {
    "benzene": "c1ccccc1", "pyridine": "c1ccncc1", "thiophene": "c1ccsc1", "furan": "c1ccoc1",
    "carboxylic acid": "CC(=O)O", "amide": "CC(=O)N", "ester": "CC(=O)OC", "urea": "NC(=O)N",
    "nitrile": "CC#N", "sulfonamide": "CS(=O)(=O)N", "trifluoromethyl": "CC(F)(F)F",
    "piperidine": "C1CCNCC1", "piperazine": "C1CNCCN1", "morpholine": "C1COCCN1",
    "phenol": "Oc1ccccc1", "aniline": "Nc1ccccc1", "chlorobenzene": "Clc1ccccc1",
    "indole": "c1ccc2[nH]ccc2c1", "biphenyl": "c1ccc(cc1)-c1ccccc1", "ketone": "CC(=O)C",
}

fg_queries = []
for name, s in FG_SMILES.items():
    m = Chem.MolFromSmiles(s)
    q = make_query(m, list(range(m.GetNumAtoms())))
    q["name"] = name
    fg_queries.append(q)

# Print a few sample queries to demonstrate
demo_queries = sample_queries(train_mols, 3, random.Random(1))
print("Sample Extracted Queries:")
for i, q in enumerate(demo_queries, 1):
    print(f"  Query #{i}: SMARTS = {q['smarts']:25s} | Atom Count = {q['graph'].num_nodes}")


# %% [markdown]
# ## 4. Non-GNN Baseline: RDKit Pattern Fingerprint
#
# Before building neural models, we implement the gold-standard classical screening baseline: **RDKit Pattern Fingerprints** (`Chem.PatternFingerprint`).
#
# ### The Substructure Containment Property
# Unlike circular fingerprints (like Morgan / ECFP), which are designed for whole-molecule similarity and cannot guarantee substructure containment, `PatternFingerprint` is specifically engineered for substructure screening:
# $$	ext{If } Q \subseteq T \implies \left( \mathbf{fp}(Q) 	ext{ bitwise-AND } \mathbf{fp}(T) 
# ight) = \mathbf{fp}(Q)$$
#
# - **Guaranteed Zero False Negatives (Recall = 1.0):** If even a single bit present in $\mathbf{fp}(Q)$ is missing in $\mathbf{fp}(T)$, the target molecule $T$ mathematically cannot contain $Q$.
# - **False Positives:** Hash collisions and combinatorial overlap can cause a molecule to pass the fingerprint filter even if the true graph topology does not match.
# - **Fast Vectorized Screening:** We pack the 2048-bit fingerprints into `uint8` arrays (`np.packbits`), allowing one query to be screened against tens of thousands of database molecules via vectorized SIMD bitwise operations in ~1 millisecond.

# %%
# ==============================================================================
# CELL 10: Pattern Fingerprint Vectorization & Precomputation
# ==============================================================================
# 1. pattern_fp_array: Computes a 2048-bit pattern fingerprint and packs it into 
#    256 uint8 bytes (2048 / 8 = 256 bytes) for ultra-fast bitwise SIMD checks.
# 2. fp_screen: Evaluates the bitwise containment condition against all DB molecules.
# 3. Precomputes fingerprints for the search database and train/val sets once offline.
# ==============================================================================

def pattern_fp_array(mol):
    """Computes a 2048-bit Pattern Fingerprint and packs it into a uint8 NumPy array."""
    fp = Chem.PatternFingerprint(mol, fpSize=CFG["FP_BITS"])
    arr = np.zeros((CFG["FP_BITS"],), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return np.packbits(arr)                         # Packs 2048 binary bits into 256 uint8 bytes

def fp_screen(q_fp, db_fp):
    """
    Vectorized bitwise containment check across the entire database:
      Condition: (db_fp & q_fp) == q_fp for all 256 bytes.
    Returns:
      Boolean NumPy mask of shape [N_db] where True indicates a candidate hit.
    """
    return np.all((db_fp & q_fp) == q_fp, axis=1)

print("Precomputing offline Pattern Fingerprints for all splits...")
t0 = time.perf_counter()
db_fp = np.stack([pattern_fp_array(m) for m in db_mols])
t_db_fp = time.perf_counter() - t0

train_fp = np.stack([pattern_fp_array(m) for m in train_mols])
val_fp   = np.stack([pattern_fp_array(m) for m in val_mols])

print(f"  - Search DB Fingerprint Matrix : {db_fp.shape} ({db_fp.nbytes / 1024**2:.2f} MB)")
print(f"  - One-time DB Fingerprint Cost : {t_db_fp:.2f} seconds")


# %% [markdown]
# ## 5. Training Pair Construction with Hard Negative Mining
#
# GNNs cannot learn substructure containment from random pairs alone. If negative pairs are randomly drawn from the database, the model quickly learns trivial shortcuts (e.g., checking whether the target simply has enough Carbon or Nitrogen atoms) rather than learning true graph topology.
#
# ### Mining "Hard Negatives"
# For every query $Q$, we construct a rich, balanced set of training pairs:
# 1. **Source Molecule (Positive, $y=1$):** The molecule from which $Q$ was extracted.
# 2. **Additional Positive ($y=1$):** Another database molecule that passes the fingerprint check and is verified by RDKit exact matching to contain $Q$.
# 3. **Hard Negative ($y=0$):** A molecule that **passes the fingerprint screening test** (so it contains all the required bits and atom counts) but **fails the exact RDKit topological check**! This forces the GNN to learn cycle connectivity, branching, and bond arrangements.
# 4. **Random Negatives ($y=0$):** Randomly sampled molecules from the general background chemical distribution.

# %%
# ==============================================================================
# CELL 12: Training Pair Generation with Hard Negatives
# ==============================================================================

def build_pairs(queries, mols, fps, rng, pool=2000):
    """
    Constructs informative (query, target_molecule, label) pairs.
    
    Includes:
      - Positive from source molecule (label = 1)
      - Verified positive from candidate pool (label = 1)
      - Hard negative that passes fingerprint filter but fails exact graph match (label = 0)
      - Random negative background samples (label = 0)
    """
    pairs = []                                      # Tuples: (query_idx, mol_idx, label)
    n = len(mols)
    mol_index = {id(m): i for i, m in enumerate(mols)}
    
    for qi, q in enumerate(queries):
        # 1. Trivial Positive: The source molecule where the query came from
        pairs.append((qi, mol_index[id(q["source"])], 1))
        
        # 2. Mine candidates passing fingerprint screen from a random sample pool
        cand = rng.sample(range(n), min(pool, n))
        qfp = pattern_fp_array(q["qmol"])
        passed = [cand[i] for i in np.where(fp_screen(qfp, fps[cand]))[0]]
        rng.shuffle(passed)
        
        pos_extra, hard_neg = None, None
        for j in passed[:30]:
            # Exact RDKit combinatorial check
            is_sub = mols[j].HasSubstructMatch(q["qmol"])
            if is_sub:
                if pos_extra is None and j != pairs[-1][1]:
                    pos_extra = j
            elif hard_neg is None:
                hard_neg = j                        # Passed fingerprint, but failed exact match!
            if pos_extra is not None and hard_neg is not None:
                break
                
        if pos_extra is not None:
            pairs.append((qi, pos_extra, 1))
        if hard_neg is not None:
            pairs.append((qi, hard_neg, 0))
            
        # 3. Two random background pairs
        for _ in range(2):
            r = rng.randrange(n)
            pairs.append((qi, r, int(mols[r].HasSubstructMatch(q["qmol"]))))
            
    return pairs

print("Constructing training and validation pairs with hard negative mining...")
t0 = time.perf_counter()
train_queries = sample_queries(train_mols, CFG["N_TRAIN_QUERIES"], random.Random(SEED + 1))
val_queries   = sample_queries(val_mols, CFG["N_VAL_QUERIES"], random.Random(SEED + 2))

train_pairs = build_pairs(train_queries, train_mols, train_fp, random.Random(SEED + 3))
val_pairs   = build_pairs(val_queries, val_mols, val_fp, random.Random(SEED + 4))
print(f"[Done] Built pairs in {time.perf_counter() - t0:.1f} s")

for name, P in [("Train", train_pairs), ("Validation", val_pairs)]:
    labels = np.array([p[2] for p in P])
    print(f"  - {name:10s}: {len(P):,} total pairs | Positive fraction: {labels.mean():.2%}")

# Precompute PyG graph representations of all train and val molecules for fast batching
print("Converting train and validation molecules to PyG graph tensors...")
train_mol_graphs = [mol_to_graph(m) for m in train_mols]
val_mol_graphs   = [mol_to_graph(m) for m in val_mols]
print("[Done] Graph conversion complete.")


# %% [markdown]
# ## 6. Graph Neural Network Architectures: Model A vs. Model B
#
# We implement a shared backbone encoder based on **Graph Isomorphism Networks (GIN)**, and compare two fundamentally distinct substructure scoring paradigms:
#
# ### The Backbone: GINEConv + Multi-Scale Readout
# - **Why GIN?** Standard Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) use mean/max aggregation, which Xu et al. (2019) mathematically proved cannot distinguish basic multiset structures. GIN uses **sum aggregation**, making it as expressive as the 1-Weisfeiler-Lehman (1-WL) graph isomorphism test.
# - **Edge Attributes with `GINEConv`:** Chemical bonds carry crucial semantics (e.g., single vs. aromatic bonds). `GINEConv` incorporates bond features directly into message passing.
# - **Multi-Scale Readout:** We perform `global_add_pool` after the input projection and after **every** message-passing layer, concatenating all pooled vectors. This allows the final embedding to perceive both localized functional motifs (1-hop bonds) and broader molecular scaffolds (4-hop rings).
#
# ---
#
# ### Comparison of the Two Model Formulations:
#
# | Dimension | Model A: Siamese GIN + MLP Classifier | Model B: NeuroMatch Order Embedder |
# | :--- | :--- | :--- |
# | **Mathematical Formulation** | $\hat{y} = \sigma\left(	ext{MLP}([\mathbf{z}_Q, \mathbf{z}_T, \mathbf{z}_T - \mathbf{z}_Q, \mathbf{z}_Q \odot \mathbf{z}_T])
# ight)$ | $\mathbf{z}_Q, \mathbf{z}_T \in \mathbb{R}_{\ge 0}^d, \quad Q \subseteq T \iff \mathbf{z}_Q \le \mathbf{z}_T$ |
# | **Scoring Metric** | Classifier logit (higher = contained) | Negative violation: $-E(Q, T) = -\|\max(0, \mathbf{z}_Q - \mathbf{z}_T)\|^2$ |
# | **Database Pre-Embedding** | ❌ Impossible (MLP must evaluate every pair) | ✅ Fully precomputable offline |
# | **Search Complexity** | $\mathcal{O}(N_{	ext{db}} 	imes 	ext{Cost}(	ext{MLP}))$ | $\mathcal{O}(N_{	ext{db}} 	imes d)$ vectorized GPU subtraction |

# %%
# ==============================================================================
# CELL 14: GIN Encoder, Model A (Siamese Classifier) & Model B (Order Embedder)
# ==============================================================================

class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) backbone with bond feature support (GINEConv).
    Uses sum aggregation and multi-scale jumping-knowledge readout.
    """
    def __init__(self, in_dim, edge_dim, hidden, out_dim, n_layers, nonneg=False):
        super().__init__()
        self.nonneg = nonneg                        # If True, applies ReLU to ensure z in R^d >= 0
        
        # Initial atom feature projection: in_dim (10) -> hidden (64)
        self.inp = nn.Linear(in_dim, hidden)
        
        # Stack of GINE message-passing layers
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            # train_eps=True learns an epsilon weighting for the central atom
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            
        # Readout MLP: pools (n_layers + 1) representations of size hidden
        # (initial input + 4 conv layers = 5 * 64 = 320 dimensions)
        self.out = nn.Sequential(
            nn.Linear(hidden * (n_layers + 1), hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, data):
        """
        Encodes a batched molecular graph into a fixed-size embedding vector.
        data.x: [total_nodes, in_dim]
        data.edge_index: [2, total_edges]
        data.edge_attr: [total_edges, edge_dim]
        data.batch: [total_nodes] mapping each node to its graph index
        Returns:
            z: [batch_size, out_dim]
        """
        h = self.inp(data.x)
        pooled = [global_add_pool(h, data.batch)]   # Readout 0: input atom features
        
        for conv in self.convs:
            # GINE message passing with residual skip connection
            h = F.relu(conv(h, data.edge_index, data.edge_attr)) + h
            pooled.append(global_add_pool(h, data.batch))
            
        # Concatenate multi-scale representations across all hops
        z = self.out(torch.cat(pooled, dim=1))
        
        # For order embeddings, clip to non-negative orthant (z >= 0)
        return F.relu(z) if self.nonneg else z

IN_DIM   = len(ELEMENTS) + 1                        # 10 atom features
EDGE_DIM = len(BOND_TYPES)                          # 4 bond features

class PairClassifier(nn.Module):
    """
    Model A: Siamese GIN + Cross-Attention Interaction Head.
    Takes [zq, zt, zt - zq, zq * zt] (4 * 64 = 256 dims) and predicts a scalar logit.
    """
    def __init__(self):
        super().__init__()
        self.enc = GINEncoder(IN_DIM, EDGE_DIM, CFG["HIDDEN"], CFG["EMB_DIM"], CFG["N_LAYERS"], nonneg=False)
        d = CFG["EMB_DIM"]
        self.head = nn.Sequential(
            nn.Linear(4 * d, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def score(self, zq, zt):
        """Scores pair containment: higher logit indicates higher likelihood of containment."""
        interaction = torch.cat([zq, zt, zt - zq, zq * zt], dim=-1)
        return self.head(interaction).squeeze(-1)

class OrderEmbedder(nn.Module):
    """
    Model B: NeuroMatch-style Order Embedding in Non-Negative Orthant R^d >= 0.
    Substructure containment is modeled as a partial order:
        Q is subgraph of T  <=>  zq <= zt element-wise in all d dimensions.
    """
    def __init__(self):
        super().__init__()
        # Enforces non-negative embeddings via ReLU activation at the output
        self.enc = GINEncoder(IN_DIM, EDGE_DIM, CFG["HIDDEN"], CFG["EMB_DIM"], CFG["N_LAYERS"], nonneg=True)

    @staticmethod
    def violation(zq, zt):
        """
        Calculates the order violation energy:
          E(q, t) = sum_i (max(0, z_q[i] - z_t[i]))^2
        If Q is contained in T (zq <= zt everywhere), violation is strictly 0.0.
        """
        return F.relu(zq - zt).pow(2).sum(-1)

    def score(self, zq, zt):
        """Higher score = lower violation = more likely contained."""
        return -self.violation(zq, zt)

print("Model architectures defined:")
print("  - Model A: Siamese GIN + Interaction MLP")
print("  - Model B: NeuroMatch Geometric Order Embedder")


# %% [markdown]
# ## 7. Model Training & Optimization
#
# We train both models using the Adam optimizer with a batch size of 256.
#
# ### Loss Functions:
# 1. **Model A (PairClassifier):** Standard Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`):
#    $$\mathcal{L}_{	ext{BCE}} = - \left[ y \log \sigma(\hat{s}) + (1 - y) \log (1 - \sigma(\hat{s})) 
# ight]$$
# 2. **Model B (OrderEmbedder):** NeuroMatch Max-Margin Contrastive Hinge Loss:
#    $$\mathcal{L}_{	ext{order}} = y \cdot E(Q, T) + (1 - y) \cdot \max(0, \gamma - E(Q, T))$$
#    - For true subgraphs ($y=1$), penalizes any positive violation energy $E(Q, T) = \|\max(0, \mathbf{z}_Q - \mathbf{z}_T)\|^2$.
#    - For non-subgraphs ($y=0$), penalizes targets that fall within margin $\gamma = 1.0$ of the query.
#
# ### Validation Metric:
# At the end of each epoch, we evaluate validation performance using **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve) computed via the rank-sum formulation (Mann-Whitney U statistic) for fast evaluation.

# %%
# ==============================================================================
# CELL 16: Mini-batch Generator, Loss Functions & Training Loop
# ==============================================================================

def make_batches(pairs, queries, mol_graphs, batch_size, shuffle, rng):
    """
    Yields mini-batches of (batched_query_graphs, batched_target_graphs, labels).
    Uses PyG Batch.from_data_list to combine individual graphs into a single GPU tensor.
    """
    idx = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        chunk = [pairs[i] for i in idx[s:s + batch_size]]
        # Combine list of PyG Data graphs into a unified Batch object
        qb = Batch.from_data_list([queries[q]["graph"] for q, _, _ in chunk]).to(DEVICE)
        tb = Batch.from_data_list([mol_graphs[t] for _, t, _ in chunk]).to(DEVICE)
        y  = torch.tensor([l for _, _, l in chunk], dtype=torch.float, device=DEVICE)
        yield qb, tb, y

def loss_fn(model, zq, zt, y):
    """Computes appropriate loss function depending on model architecture."""
    if isinstance(model, OrderEmbedder):
        # NeuroMatch max-margin hinge loss
        e = model.violation(zq, zt)
        return (y * e + (1 - y) * F.relu(CFG["MARGIN"] - e)).mean()
    # Binary cross-entropy with logits for Siamese classifier
    return F.binary_cross_entropy_with_logits(model.score(zq, zt), y)

@torch.no_grad()
def scores_on_pairs(model, pairs, queries, mol_graphs):
    """Evaluates model scores across a dataset of pairs without computing gradients."""
    model.eval()
    out = []
    for qb, tb, _ in make_batches(pairs, queries, mol_graphs, 1024, False, None):
        out.append(model.score(model.enc(qb), model.enc(tb)).cpu())
    return torch.cat(out).numpy()

def roc_auc(y, s):
    """
    Computes Receiver Operating Characteristic Area Under the Curve (ROC-AUC)
    using the exact rank-sum formulation (Mann-Whitney U test).
    """
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def train(model, name):
    """Executes model training across all epochs and logs loss and validation ROC-AUC."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    rng = random.Random(SEED)
    y_val = np.array([p[2] for p in val_pairs])
    history = []
    
    print(f"\n================ Starting Training: {name} ================")
    for ep in range(1, CFG["EPOCHS"] + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss, num_batches = 0.0, 0
        
        for qb, tb, y in make_batches(train_pairs, train_queries, train_mol_graphs, CFG["BATCH"], True, rng):
            # Forward pass: encode queries and targets
            zq = model.enc(qb)
            zt = model.enc(tb)
            loss = loss_fn(model, zq, zt, y)
            
            # Backpropagation & Adam weight update
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        
        # Validation evaluation
        val_scores = scores_on_pairs(model, val_pairs, val_queries, val_mol_graphs)
        auc = roc_auc(y_val, val_scores)
        elapsed = time.perf_counter() - t0
        
        history.append(dict(epoch=ep, loss=avg_loss, val_auc=auc))
        print(f"[{name}] Epoch {ep:2d}/{CFG['EPOCHS']:2d} | Loss: {avg_loss:.4f} | Val ROC-AUC: {auc:.4f} | Time: {elapsed:.1f}s")
        
    return pd.DataFrame(history)

# Train Model A (Siamese Classifier)
model_A = PairClassifier()
hist_A  = train(model_A, "Model A (GIN + MLP BCE)")

# Train Model B (NeuroMatch Order Embedder)
model_B = OrderEmbedder()
hist_B  = train(model_B, "Model B (GIN Order Embedding)")


# %%
# ==============================================================================
# CELL 17: Training Loss & Validation ROC-AUC Curves
# ==============================================================================
# Plots convergence behavior for Model A and Model B side-by-side.
# Saves plot to results/training_curves.png.
# ==============================================================================

fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=120)

for h, label, color in [(hist_A, "Model A: GIN + MLP (BCE)", "#9333ea"), 
                        (hist_B, "Model B: Order Embedding", "#2563eb")]:
    # Left subplot: Training loss progression
    ax[0].plot(h.epoch, h.loss, marker="o", markersize=4, label=label, color=color, lw=1.8)
    # Right subplot: Validation ROC-AUC progression
    ax[1].plot(h.epoch, h.val_auc, marker="s", markersize=4, label=label, color=color, lw=1.8)

ax[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax[0].set_ylabel("Training Loss", fontsize=11, fontweight="bold")
ax[0].set_title("Training Loss Convergence", fontsize=12, fontweight="bold")
ax[0].grid(True, ls="--", alpha=0.5)
ax[0].legend(fontsize=9.5)

ax[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax[1].set_ylabel("Validation ROC-AUC", fontsize=11, fontweight="bold")
ax[1].set_title("Validation Discrimination Performance", fontsize=12, fontweight="bold")
ax[1].grid(True, ls="--", alpha=0.5)
ax[1].legend(fontsize=9.5)

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
plt.show()
print("[Saved] Training curves saved to 'results/training_curves.png'")


# %% [markdown]
# ## 8. Decision Threshold Tuning on a Realistic Validation Search
#
# A critical subtlety often overlooked in chemical machine learning is the **Class Imbalance Problem**:
# - During training, pairs were roughly balanced (50% positive, 50% negative).
# - In a real chemical search, however, **$< 5\%$ of molecules** in a database contain any given query!
# - If a decision threshold is tuned on balanced pairs, it will generate an unbearable flood of false positives during a real search.
#
# ### Two Distinct Threshold Calibration Objectives:
# 1. **Optimal F1 Threshold (`thr_f1`):** Calibrated for standalone GNN retrieval. Maximizes harmonic mean of precision and recall ($F_1$).
# 2. **High-Recall Cascade Threshold (`thr_rec`):** Calibrated for the Two-Stage Cascade. We choose a looser threshold guaranteeing **$\ge 98\%$ recall** on true hits, ensuring that virtually no valid candidate is discarded before RDKit verification.

# %%
# ==============================================================================
# CELL 19: Decision Threshold Calibration on Realistic Validation Search
# ==============================================================================

# Run a validation search: evaluate validation queries against all validation molecules
N_TUNE = min(200, len(val_queries))
tune_q = val_queries[:N_TUNE]

# Ground-truth binary containment matrix Y_val: shape [N_tune, N_val]
Y_val = np.array([[m.HasSubstructMatch(q["qmol"]) for m in val_mols] for q in tune_q])
print(f"Validation Search Simulation: {N_TUNE} queries x {len(val_mols)} molecules")
print(f"Realistic Database Hit Rate : {Y_val.mean():.2%} (severe class imbalance)")

@torch.no_grad()
def score_matrix(model, queries, z_targets):
    """Computes pairwise model prediction scores between queries and target embeddings."""
    model.eval()
    # Batch encode all queries: shape [N_q, d]
    zq = model.enc(Batch.from_data_list([q["graph"] for q in queries]).to(DEVICE))
    scores = []
    for i in range(len(queries)):
        # Expand query embedding across all targets and score
        s = model.score(zq[i].expand(z_targets.shape[0], -1), z_targets).cpu()
        scores.append(s)
    return torch.stack(scores).numpy()

def pick_thresholds(model):
    """
    Sweeps candidate score thresholds to determine:
    1. thr_f1: Threshold that maximizes F1 score.
    2. thr_rec: Threshold that retains >= 98% of true positives (for cascade).
    """
    model.eval()
    with torch.no_grad():
        zval = model.enc(Batch.from_data_list(val_mol_graphs).to(DEVICE))
    S = score_matrix(model, tune_q, zval)
    
    best_f1, thr_f1 = -1.0, None
    # Sweep through quantiles of predicted score distribution
    for t in np.unique(np.quantile(S, np.linspace(0.3, 0.9999, 500))):
        pred = S >= t
        tp = (pred & Y_val).sum()
        fp = (pred & ~Y_val).sum()
        fn = (~pred & Y_val).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, thr_f1 = f1, t
            
    # High-recall threshold: the 2nd percentile of scores on true positive pairs
    # (meaning 98% of true matches score above this value)
    thr_rec = np.quantile(S[Y_val], 0.02)
    return float(thr_f1), float(thr_rec), float(best_f1)

thr_A, _, f1A = pick_thresholds(model_A)
thr_B, thr_B_casc, f1B = pick_thresholds(model_B)

print(f"\nCalibrated Thresholds:")
print(f"  - Model A Standalone Threshold  : {thr_A:.4f} (Validation F1 = {f1A:.3f})")
print(f"  - Model B Standalone Threshold  : {thr_B:.4f} (Validation F1 = {f1B:.3f})")
print(f"  - Model B 98% Recall Threshold  : {thr_B_casc:.4f} (Used for Two-Stage Cascade)")


# %% [markdown]
# ## 9. Full Retrieval Benchmark on the Held-Out Chemical Database
#
# In this section, we conduct our primary quantitative experiment: evaluating all retrieval methods on the held-out search database.
#
# ### Evaluation Protocol:
# - **Test Query Set:** Consists of $N_{	ext{test}}$ sampled subgraph queries extracted from the independent `qsrc_mols` pool, plus the 20 medicinal functional group queries.
# - **Hardware Timing:** We use `torch.cuda.synchronize()` before and after GPU operations to ensure precise wall-clock latency measurement without asynchronous queuing distortions.
# - **One-time vs. Per-Query Costs:**
#   - One-time costs (computing database fingerprints or GNN database embeddings) are precomputed offline.
#   - Per-query latency measures the elapsed time from query arrival to final result set return.

# %%
# ==============================================================================
# CELL 21: Database Pre-embedding & Per-Query Benchmark Execution
# ==============================================================================

def sync():
    """Ensures all pending asynchronous CUDA GPU kernels have completed."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def embed_graphs(model, graphs, bs=2048):
    """Encodes a large list of PyG graphs in mini-batches to prevent GPU OOM."""
    model.eval()
    out = []
    for s in range(0, len(graphs), bs):
        batch = Batch.from_data_list(graphs[s:s + bs]).to(DEVICE)
        out.append(model.enc(batch))
    return torch.cat(out)

# Convert all search database molecules to PyG graphs
print(f"Converting {len(db_mols):,} search database molecules into graph tensors...")
db_graphs = [mol_to_graph(m) for m in db_mols]

# Precompute offline embeddings for Model A and Model B
emb_time = {}
for name, model in [("A", model_A), ("B", model_B)]:
    sync()
    t0 = time.perf_counter()
    z = embed_graphs(model, db_graphs)
    sync()
    emb_time[name] = time.perf_counter() - t0
    if name == "A":
        zdb_A = z
    else:
        zdb_B = z

print(f"Offline Database Pre-embedding Latency:")
print(f"  - Model A DB Embedding Time : {emb_time['A']:.2f} s")
print(f"  - Model B DB Embedding Time : {emb_time['B']:.2f} s")

@torch.no_grad()
def gnn_hits(model, zdb, q, thr):
    """Screens all database embeddings against a query embedding on GPU."""
    zq = model.enc(Batch.from_data_list([q["graph"]]).to(DEVICE))
    s = model.score(zq.expand(zdb.shape[0], -1), zdb)
    return (s >= thr).nonzero().squeeze(-1).cpu().numpy()

# Prepare test query suite (independent sampled subgraphs + curated functional groups)
test_queries = sample_queries(qsrc_mols, CFG["N_TEST_QUERIES"], random.Random(SEED + 5))
for q in test_queries:
    q["name"] = "sampled"
all_test = test_queries + fg_queries
N = len(db_mols)

print(f"\nEvaluating {len(all_test)} test queries across {N:,} database molecules...")
benchmark_rows = []

for qi, q in enumerate(all_test):
    qm = q["qmol"]
    r = dict(query=q["smarts"], kind=q["name"], q_atoms=q["graph"].num_nodes)

    # 1. RDKit Exact Ground Truth (Brute Force Combinatorial Matching)
    t0 = time.perf_counter()
    truth = np.array([m.HasSubstructMatch(qm) for m in db_mols])
    r["t_rdkit_brute"] = time.perf_counter() - t0
    truth_set = set(np.where(truth)[0])
    r["truth"] = truth_set
    r["n_true"] = len(truth_set)

    # 2. Classical Pattern Fingerprint Screen Only
    t0 = time.perf_counter()
    cand = np.where(fp_screen(pattern_fp_array(qm), db_fp))[0]
    r["t_fp_screen"] = time.perf_counter() - t0
    r["fp_screen"] = set(cand.tolist())

    # 3. Classical Two-Stage Baseline (Pattern FP Screen + RDKit Verification)
    t0 = time.perf_counter()
    verified = [i for i in cand if db_mols[i].HasSubstructMatch(qm)]
    r["t_fp_rdkit"] = r["t_fp_screen"] + (time.perf_counter() - t0)
    r["fp_rdkit"] = set(verified)

    # 4. Neural Standalone Retrieval (GNN A, GNN B, and Cascade Shortlist)
    for key, model, zdb, thr in [
        ("gnn_A",   model_A, zdb_A, thr_A),
        ("gnn_B",   model_B, zdb_B, thr_B),
        ("short_B", model_B, zdb_B, thr_B_casc)
    ]:
        sync()
        t0 = time.perf_counter()
        hits = gnn_hits(model, zdb, q, thr)
        sync()
        r["t_" + key] = time.perf_counter() - t0
        r[key] = set(hits.tolist())

    # 5. Two-Stage Neural Cascade (GNN B Shortlist + RDKit Verification)
    t0 = time.perf_counter()
    cascade_hits = {i for i in r["short_B"] if db_mols[i].HasSubstructMatch(qm)}
    r["t_cascade_B"] = r["t_short_B"] + (time.perf_counter() - t0)
    r["cascade_B"] = cascade_hits

    benchmark_rows.append(r)
    if (qi + 1) % 50 == 0 or (qi + 1) == len(all_test):
        print(f"  --> Progress: {qi + 1:3d}/{len(all_test)} queries evaluated...")

print("[Done] Benchmark evaluation finished.")


# %%
# ==============================================================================
# CELL 22: Benchmark Summary Metrics & CSV Export
# ==============================================================================

# Define evaluated methods and their corresponding dict keys
METHODS = [
    ("RDKit exact, brute force",        "truth",      "t_rdkit_brute"),
    ("Pattern FP screen only",          "fp_screen",  "t_fp_screen"),
    ("Pattern FP screen + RDKit",       "fp_rdkit",   "t_fp_rdkit"),
    ("GNN A: GIN + MLP",                "gnn_A",      "t_gnn_A"),
    ("GNN B: order embedding",          "gnn_B",      "t_gnn_B"),
    ("Cascade: GNN B shortlist + RDKit","cascade_B",  "t_cascade_B")
]

def summarise(rows_subset):
    """Computes micro-averaged precision, recall, F1, latency, and speedup."""
    out = []
    for label, key, tkey in METHODS:
        # Micro-aggregated true positives, false positives, and false negatives
        tp = sum(len(r[key] & r["truth"]) for r in rows_subset)
        fp = sum(len(r[key] - r["truth"]) for r in rows_subset)
        fn = sum(len(r["truth"] - r[key]) for r in rows_subset)
        
        p  = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        ms = 1000.0 * np.mean([r[tkey] for r in rows_subset])
        avg_ret = np.mean([len(r[key]) for r in rows_subset])
        
        out.append(dict(
            method=label,
            precision=round(p, 4),
            recall=round(rc, 4),
            f1=round(f1, 4),
            ms_per_query=round(ms, 2),
            avg_returned=round(avg_ret, 1)
        ))
        
    df = pd.DataFrame(out)
    base_time = df.loc[0, "ms_per_query"]
    df["speedup_vs_brute"] = (base_time / df["ms_per_query"]).round(1)
    return df

res_all     = summarise(benchmark_rows)
res_sampled = summarise([r for r in benchmark_rows if r["kind"] == "sampled"])
res_fg      = summarise([r for r in benchmark_rows if r["kind"] != "sampled"])

print(f"\n=================== BENCHMARK: ALL TEST QUERIES ({len(all_test)}) ===================")
print(res_all.to_string(index=False))

print(f"\n================ BENCHMARK: SAMPLED SUBGRAPHS ({len(test_queries)}) ================")
print(res_sampled.to_string(index=False))

print(f"\n============== BENCHMARK: MEDICINAL FUNCTIONAL GROUPS ({len(fg_queries)}) ==============")
print(res_fg.to_string(index=False))

# Export benchmark CSVs to disk
res_all.to_csv("results/benchmark_all.csv", index=False)
res_sampled.to_csv("results/benchmark_sampled.csv", index=False)
res_fg.to_csv("results/benchmark_functional_groups.csv", index=False)
print("\n[Saved] Benchmark tables exported to './results/'")


# %% [markdown]
# ## 10. Accuracy vs. Speed Trade-offs: The Pareto Frontier
#
# In engineering system design, the ideal algorithm resides in the **top-left corner** of a Latency vs. Accuracy plot: **lowest latency (fastest)** and **highest F1 score (most accurate)**.
#
# ### The Central Technical Insight: The "Verification Bottleneck"
# Our empirical results reveal a critical scientific finding regarding neural substructure retrieval:
# - **GNN B Alone is Ultra-Fast:** Vector screening of order embeddings on GPU takes only **2.79 ms per query** ($11.4	imes$ faster than brute-force RDKit).
# - **The Cost of High Recall:** To prevent missing valid drug candidates, the cascade threshold must be relaxed to maintain $\ge 98\%$ recall.
# - **The False Positive Explosion:** At this threshold, GNN B admits **1,414 candidate molecules** per query—nearly double the 703 candidates admitted by classical Pattern Fingerprints.
# - **The Bottleneck:** Because combinatorial verification in RDKit requires $pprox 0.012	ext{ ms}$ per candidate molecule, verifying those extra false positives consumes **$17.4	ext{ ms}$ of CPU time**!
# - As a result, the end-to-end latency of the GNN Cascade drops to **20.21 ms** ($1.6	imes$ speedup), whereas the classical Pattern Fingerprint baseline achieves **5.61 ms** ($5.7	imes$ speedup) with 100% precision and 100% recall.

# %%
# ==============================================================================
# CELL 24: Latency vs. F1 Pareto Frontier & Threshold Sensitivity Curve
# ==============================================================================
# Generates high-resolution trade-off figures with individual callout boxes
# and pointer arrows to ensure zero text overlap.
# Saves figure to results/tradeoff.png.
# ==============================================================================

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)
plt.subplots_adjust(wspace=0.28, left=0.07, right=0.96, top=0.88, bottom=0.14)

# Custom visual styling for each evaluated retrieval method
method_styles = {
    "Pattern FP screen only": {
        "color": "#ea580c", "marker": "s", "size": 85, 
        "text_offset": (15, 12), "label": "Pattern FP screen only"
    },
    "GNN B: order embedding": {
        "color": "#2563eb", "marker": "p", "size": 95, 
        "text_offset": (18, -12), "label": "GNN B: Order Embedding alone"
    },
    "GNN A: GIN + MLP": {
        "color": "#9333ea", "marker": "^", "size": 95, 
        "text_offset": (18, 8), "label": "GNN A: GIN + MLP"
    },
    "Pattern FP screen + RDKit": {
        "color": "#16a34a", "marker": "o", "size": 105, 
        "text_offset": (-150, 14), "label": "Pattern FP + RDKit (Baseline)"
    },
    "Cascade: GNN B shortlist + RDKit": {
        "color": "#0284c7", "marker": "*", "size": 160, 
        "text_offset": (-125, -42), "label": "Cascade: GNN B + RDKit"
    },
    "RDKit exact, brute force": {
        "color": "#dc2626", "marker": "D", "size": 85, 
        "text_offset": (14, -20), "label": "RDKit exact (Brute Force)"
    }
}

# (a) Left Subplot: Latency vs. F1 Pareto Frontier
for _, row in res_all.iterrows():
    m = row["method"]
    cfg = method_styles.get(m, {"color": "black", "marker": "o", "size": 60, "text_offset": (10, 10), "label": m})
    x = row["ms_per_query"]
    y = row["f1"]
    
    # Plot method marker
    ax[0].scatter(x, y, color=cfg["color"], marker=cfg["marker"], s=cfg["size"], edgecolor="black", linewidth=0.8, zorder=5)
    
    # Add annotated callout box with pointer arrow
    dx, dy = cfg["text_offset"]
    ax[0].annotate(
        f"{cfg['label']}\n({x:.1f} ms, F1={y:.2f})",
        xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=8, fontweight="bold" if "Cascade" in m or "Baseline" in cfg["label"] else "normal",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=cfg["color"], lw=1.2, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1.1, shrinkA=3, shrinkB=4),
        zorder=6
    )

ax[0].set_xscale("log")
ax[0].set_xlim(0.7, 85)
ax[0].set_ylim(0.30, 1.10)
ax[0].set_xlabel("Time per Query (ms, log scale)", fontsize=10.5, fontweight="bold", labelpad=6)
ax[0].set_ylabel("F1 Score vs. Exact Ground Truth", fontsize=10.5, fontweight="bold", labelpad=6)
ax[0].set_title(f"(a) Retrieval Accuracy vs. Latency Pareto Frontier ({N} Molecules)", fontsize=11, fontweight="bold", pad=10)
ax[0].grid(True, which="both", ls="--", lw=0.5, alpha=0.45)
ax[0].tick_params(labelsize=9)

# (b) Right Subplot: Precision-Recall Curve of Order Embedding Model across Thresholds
sub = [r for r in benchmark_rows if r["kind"] == "sampled"][:100]
zq_all = embed_graphs(model_B, [all_test[benchmark_rows.index(r)]["graph"] for r in sub])
P, R = [], []
with torch.no_grad():
    S = torch.stack([model_B.score(zq_all[i].expand(N, -1), zdb_B).cpu() for i in range(len(sub))]).numpy()
Y = np.zeros_like(S, dtype=bool)
for i, r in enumerate(sub):
    Y[i, list(r["truth"])] = True
for t in np.quantile(S, np.linspace(0.5, 0.9999, 60)):
    pred = S >= t
    tp = (pred & Y).sum()
    P.append(tp / max(pred.sum(), 1))
    R.append(tp / max(Y.sum(), 1))

ax[1].plot(R, P, marker="o", markersize=3.5, color="#1e40af", lw=1.8, label="Order Embedding PR Curve")
ax[1].fill_between(R, P, alpha=0.12, color="#3b82f6")
ax[1].axvline(0.98, color="#dc2626", ls=":", lw=1.2, alpha=0.8, label="98% Recall Operating Line")
ax[1].set_xlim(0.56, 1.02)
ax[1].set_ylim(0.16, 0.285)
ax[1].set_xlabel("Recall (True Positives / All Subgraph Matches)", fontsize=10.5, fontweight="bold", labelpad=6)
ax[1].set_ylabel("Precision (True Positives / Shortlisted)", fontsize=10.5, fontweight="bold", labelpad=6)
ax[1].set_title("(b) GNN B Order Embeddings: Threshold Sensitivity", fontsize=11, fontweight="bold", pad=10)
ax[1].grid(True, ls="--", lw=0.5, alpha=0.45)
ax[1].tick_params(labelsize=9)
ax[1].legend(loc="upper right", fontsize=8.5, framealpha=0.9)

plt.savefig("results/tradeoff.png", dpi=300, bbox_inches="tight")
plt.show()
print("[Saved] Clean trade-off plot saved to 'results/tradeoff.png'")


# %% [markdown]
# ## 11. Error Analysis: Where Does the GNN Fail?
#
# To understand the fundamental limitations of the GNN representations, we stratify query retrieval accuracy by **query size** (number of atoms in $Q$).
#
# ### Why Recall Degrades on Larger Queries:
# - **Small Queries (4–5 atoms):** The GNN achieves high recall (**$83.4\%–83.7\%$**). Local atomic motifs fall cleanly inside the 4-layer receptive field.
# - **Large Queries (10–12 atoms):** Recall drops precipitously to **$28.2\%–44.1\%$**.
# - **The Graph Theoretical Explanation:**
#   1. **Receptive Field Bound:** A 4-layer GNN can only propagate information across 4 hops. A 10-atom ring system or extended bicyclic fragment has a topological diameter exceeding 4 hops, meaning no single node state can perceive the global cycle structure.
#   2. **Whole-Graph Readout Loss:** `global_add_pool` collapses the entire molecule into a single vector $\mathbf{z} \in \mathbb{R}^{64}$. While effective for small fragments, whole-graph pooling discards localized node-to-node correspondence, causing order violations on large complex topologies.

# %%
# ==============================================================================
# CELL 26: Stratified Performance Breakdown by Query Size & Functional Groups
# ==============================================================================

# Group sampled queries by atom count to analyze size-dependent performance
size_rows = []
for r in [r for r in benchmark_rows if r["kind"] == "sampled"]:
    tp = len(r["gnn_B"] & r["truth"])
    size_rows.append(dict(
        q_atoms=r["q_atoms"],
        recall=tp / max(len(r["truth"]), 1),
        precision=tp / max(len(r["gnn_B"]), 1),
        n_true=r["n_true"]
    ))

by_size = pd.DataFrame(size_rows).groupby("q_atoms").mean().round(3)
print("=== Performance Breakdown by Query Size (Order Embedding GNN B) ===")
print(by_size)
by_size.to_csv("results/order_model_by_query_size.csv")

print("\n=== Performance on Real-World Functional Group Queries ===")
fg_tab = pd.DataFrame([
    dict(
        functional_group=r["kind"],
        true_db_matches=r["n_true"],
        gnn_shortlisted=len(r["gnn_B"]),
        gnn_correct_matches=len(r["gnn_B"] & r["truth"]),
        gnn_precision=round(len(r["gnn_B"] & r["truth"]) / max(len(r["gnn_B"]), 1), 3),
        gnn_recall=round(len(r["gnn_B"] & r["truth"]) / max(r["n_true"], 1), 3)
    )
    for r in benchmark_rows if r["kind"] != "sampled"
])
print(fg_tab.to_string(index=False))
fg_tab.to_csv("results/order_model_functional_groups.csv", index=False)


# %%
# ==============================================================================
# CELL 27: Model Checkpointing & Configuration Persistence
# ==============================================================================
# Saves trained neural network weights, calibrated thresholds, and runtime
# configuration into the results directory for persistent reproducibility.
# ==============================================================================

# Save PyTorch model state dicts and calibrated thresholds
torch.save({
    "model_A": model_A.state_dict(),
    "model_B": model_B.state_dict(),
    "cfg": CFG,
    "thr_A": thr_A,
    "thr_B": thr_B,
    "thr_B_casc": thr_B_casc
}, "results/models.pt")

# Save experiment configuration and hardware metadata as JSON
config_export = dict(
    CFG,
    device=str(DEVICE),
    db_fp_seconds=t_db_fp,
    db_embed_seconds=emb_time
)
with open("results/config.json", "w") as f:
    json.dump(config_export, f, indent=2)

print("=" * 70)
print("[SUCCESS] All experiment artifacts successfully saved in './results':")
print("  - Trained Model Checkpoints : results/models.pt")
print("  - Configuration & Timing    : results/config.json")
print("  - Benchmark Result Tables   : results/benchmark_all.csv")
print("  - Query Size Breakdown CSV  : results/order_model_by_query_size.csv")
print("  - Training Curves Plot      : results/training_curves.png")
print("  - Accuracy-Speed Pareto Plot: results/tradeoff.png")
print("=" * 70)


