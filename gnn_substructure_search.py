# %% [markdown]
# # GNN Molecular Substructure Search (CS 6012 Open Project)
#
# **Question:** can a learned GNN retrieve most molecules that contain a query pattern Q,
# substantially faster than RDKit's exact substructure matcher?
#
# **Pipeline**
# 1. Data: molecules from ZINC250k, split into train / validation / search-database / held-out query sources.
# 2. Queries: connected subgraphs sampled from molecules, plus common functional groups.
# 3. Ground truth: RDKit exact matching (`HasSubstructMatch`).
# 4. Non-GNN baseline: RDKit pattern fingerprint screen (guarantees no false negatives).
# 5. GNN model A: Siamese GIN encoder + MLP classifier (BCE loss).
# 6. GNN model B (extension): NeuroMatch-style order embeddings, `z_Q <= z_T` element-wise.
# 7. Cascade (extension): GNN shortlist, then RDKit verifies only the shortlist.
# 8. Benchmark: precision / recall / F1 and time per query for every method.
#
# **How to run:** Runtime > Change runtime type > T4 GPU, then Runtime > Run all.
# Set `QUICK = True` in the config cell for a fast smoke test (about 5 minutes).

# %%
# Install libraries (Colab already has PyTorch)
import importlib.util, subprocess, sys
for pkg, mod in [("rdkit", "rdkit"), ("torch_geometric", "torch_geometric")]:
    if importlib.util.find_spec(mod) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# %%
import os, time, random, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_add_pool
from rdkit import Chem, RDLogger, DataStructs
RDLogger.DisableLog("rdApp.*")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "| torch", torch.__version__)

# %% [markdown]
# ## 1. Configuration
# All sizes are here. `QUICK = True` is a small smoke test; `QUICK = False` is the full experiment.

# %%
QUICK = False
SEED = 0

if QUICK:
    CFG = dict(N_TRAIN_MOLS=4000, N_VAL_MOLS=500, N_DB=5000, N_QSRC=200,
               N_TRAIN_QUERIES=4000, N_VAL_QUERIES=500, N_TEST_QUERIES=100,
               EPOCHS=4, BATCH=256)
else:
    CFG = dict(N_TRAIN_MOLS=40000, N_VAL_MOLS=4000, N_DB=50000, N_QSRC=2000,
               N_TRAIN_QUERIES=40000, N_VAL_QUERIES=3000, N_TEST_QUERIES=300,
               EPOCHS=20, BATCH=256)
CFG.update(QUERY_MIN_ATOMS=4, QUERY_MAX_ATOMS=12, HIDDEN=64, EMB_DIM=64,
           N_LAYERS=4, LR=1e-3, MARGIN=1.0, FP_BITS=2048)
print(CFG)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("results", exist_ok=True)

# %% [markdown]
# ## 2. Load molecules (ZINC250k) and split
# Splits are disjoint, so the search database contains molecules the GNN never saw in training.

# %%
ZINC_URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
            "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
if not os.path.exists("zinc250k.csv"):
    import urllib.request
    urllib.request.urlretrieve(ZINC_URL, "zinc250k.csv")

smiles_all = [s.strip() for s in pd.read_csv("zinc250k.csv")["smiles"]]
rng = random.Random(SEED)
rng.shuffle(smiles_all)

need = CFG["N_TRAIN_MOLS"] + CFG["N_VAL_MOLS"] + CFG["N_DB"] + CFG["N_QSRC"]
mols_all = []
for s in smiles_all:
    m = Chem.MolFromSmiles(s)
    if m is not None:
        mols_all.append(m)
    if len(mols_all) == need:
        break

a = CFG["N_TRAIN_MOLS"]; b = a + CFG["N_VAL_MOLS"]; c = b + CFG["N_DB"]
train_mols, val_mols, db_mols, qsrc_mols = mols_all[:a], mols_all[a:b], mols_all[b:c], mols_all[c:]
print(f"train {len(train_mols)} | val {len(val_mols)} | search DB {len(db_mols)} | query sources {len(qsrc_mols)}")
print("Mean atoms per DB molecule:", np.mean([m.GetNumAtoms() for m in db_mols]).round(1))

# %% [markdown]
# ## 3. Graph features and queries
#
# A query is a connected set of atoms taken from a molecule. It is written as SMARTS with
# element (`[#6]`) and bond type (`-`, `=`, `#`, `:`), and the GNN sees exactly the same
# information: element one-hot per atom, bond type one-hot per edge. Keeping the features and the
# matching rules identical means a true subgraph always has features that are a subset of the target's.

# %%
ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53]          # C N O F P S Cl Br I (+ "other")
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

def atom_feat(atom):
    v = [0.0] * (len(ELEMENTS) + 1)
    z = atom.GetAtomicNum()
    v[ELEMENTS.index(z) if z in ELEMENTS else -1] = 1.0
    return v

def bond_feat(bond):
    v = [0.0] * len(BOND_TYPES)
    bt = bond.GetBondType()
    v[BOND_TYPES.index(bt) if bt in BOND_TYPES else 0] = 1.0
    return v

def mol_to_graph(mol, atom_ids=None):
    """PyG graph of a molecule, or of the subgraph induced by atom_ids."""
    if atom_ids is None:
        atom_ids = list(range(mol.GetNumAtoms()))
    pos = {a: i for i, a in enumerate(atom_ids)}
    x = torch.tensor([atom_feat(mol.GetAtomWithIdx(a)) for a in atom_ids], dtype=torch.float)
    src, dst, ea = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in pos and j in pos:
            f = bond_feat(bond)
            src += [pos[i], pos[j]]; dst += [pos[j], pos[i]]; ea += [f, f]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(ea, dtype=torch.float) if ea else torch.zeros((0, len(BOND_TYPES)))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def sample_connected_atoms(mol, k, rng):
    """Grow a random connected atom set of size k (breadth-first style)."""
    sel = {rng.randrange(mol.GetNumAtoms())}
    while len(sel) < k:
        frontier = [n.GetIdx() for a in sel for n in mol.GetAtomWithIdx(a).GetNeighbors()
                    if n.GetIdx() not in sel]
        if not frontier:
            break
        sel.add(rng.choice(frontier))
    return sorted(sel)

def make_query(mol, atom_ids):
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)      # otherwise SMARTS gets [C@@H], which also constrains H count
    smarts = Chem.MolFragmentToSmarts(flat, atomsToUse=atom_ids)
    return dict(smarts=smarts, qmol=Chem.MolFromSmarts(smarts), graph=mol_to_graph(mol, atom_ids))

def sample_queries(mols, n, rng):
    out = []
    while len(out) < n:
        m = rng.choice(mols)
        k = rng.randint(CFG["QUERY_MIN_ATOMS"], min(CFG["QUERY_MAX_ATOMS"], m.GetNumAtoms()))
        q = make_query(m, sample_connected_atoms(m, k, rng))
        q["source"] = m
        out.append(q)
    return out

# Hand-picked functional groups (realistic chemist queries), converted to the same SMARTS form
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
    q = make_query(m, list(range(m.GetNumAtoms()))); q["name"] = name
    fg_queries.append(q)

ex = sample_queries(train_mols, 3, random.Random(1))
for q in ex:
    print("example query:", q["smarts"], "| atoms:", q["graph"].num_nodes)

# %% [markdown]
# ## 4. Non-GNN baseline: RDKit pattern fingerprint
# `Chem.PatternFingerprint` is designed for substructure screening: if Q is a substructure of T,
# every bit set in fp(Q) is also set in fp(T). So the screen never misses a true hit (recall = 1),
# but it lets through false positives. Note: Morgan/ECFP fingerprints do NOT have this guarantee.
# Fingerprints are packed into uint8 arrays so one query is screened against the whole DB with numpy.

# %%
def pattern_fp_array(mol):
    fp = Chem.PatternFingerprint(mol, fpSize=CFG["FP_BITS"])
    arr = np.zeros((CFG["FP_BITS"],), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return np.packbits(arr)

def fp_screen(q_fp, db_fp):
    return np.all((db_fp & q_fp) == q_fp, axis=1)

t0 = time.perf_counter()
db_fp = np.stack([pattern_fp_array(m) for m in db_mols])
t_db_fp = time.perf_counter() - t0
train_fp = np.stack([pattern_fp_array(m) for m in train_mols])
val_fp = np.stack([pattern_fp_array(m) for m in val_mols])
print(f"DB fingerprints: {db_fp.shape}, one-time cost {t_db_fp:.1f} s")

# %% [markdown]
# ## 5. Training pairs with RDKit labels
# For every query we build up to five (query, target, label) pairs:
# - the molecule it came from (positive),
# - another molecule that passes the fingerprint screen and the exact check (positive, if one exists),
# - a "hard negative": passes the fingerprint screen but fails the exact check,
# - two random molecules (labelled by the exact check, usually negative).
#
# Hard negatives force the GNN to learn connectivity, not just "which atoms are present".

# %%
def build_pairs(queries, mols, fps, rng, pool=2000):
    graphs = {}
    pairs = []          # (query_idx, mol_idx, label)
    n = len(mols)
    mol_index = {id(m): i for i, m in enumerate(mols)}
    for qi, q in enumerate(queries):
        pairs.append((qi, mol_index[id(q["source"])], 1))
        cand = rng.sample(range(n), min(pool, n))
        qfp = pattern_fp_array(q["qmol"])
        passed = [cand[i] for i in np.where(fp_screen(qfp, fps[cand]))[0]]
        rng.shuffle(passed)
        pos_extra, hard_neg = None, None
        for j in passed[:30]:
            if mols[j].HasSubstructMatch(q["qmol"]):
                if pos_extra is None and j != pairs[-1][1]:
                    pos_extra = j
            elif hard_neg is None:
                hard_neg = j
            if pos_extra is not None and hard_neg is not None:
                break
        if pos_extra is not None:
            pairs.append((qi, pos_extra, 1))
        if hard_neg is not None:
            pairs.append((qi, hard_neg, 0))
        for _ in range(2):
            r = rng.randrange(n)
            pairs.append((qi, r, int(mols[r].HasSubstructMatch(q["qmol"]))))
    return pairs

t0 = time.perf_counter()
train_queries = sample_queries(train_mols, CFG["N_TRAIN_QUERIES"], random.Random(SEED + 1))
val_queries = sample_queries(val_mols, CFG["N_VAL_QUERIES"], random.Random(SEED + 2))
train_pairs = build_pairs(train_queries, train_mols, train_fp, random.Random(SEED + 3))
val_pairs = build_pairs(val_queries, val_mols, val_fp, random.Random(SEED + 4))
print(f"Built pairs in {time.perf_counter() - t0:.1f} s")
for name, P in [("train", train_pairs), ("val", val_pairs)]:
    labels = np.array([p[2] for p in P])
    print(f"{name}: {len(P)} pairs, positive fraction {labels.mean():.2f}")

train_mol_graphs = [mol_to_graph(m) for m in train_mols]
val_mol_graphs = [mol_to_graph(m) for m in val_mols]

# %% [markdown]
# ## 6. GNN encoder (GIN with bond features)
# GIN uses sum aggregation, which Xu et al. (2019) show is as expressive as the 1-WL test.
# `GINEConv` adds bond-type information to each message. Following NeuroMatch, we sum-pool the
# node states of every layer and concatenate them, so the embedding sees local and wider context.

# %%
class GINEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden, out_dim, n_layers, nonneg=False):
        super().__init__()
        self.nonneg = nonneg
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
        self.out = nn.Sequential(nn.Linear(hidden * (n_layers + 1), hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, data):
        h = self.inp(data.x)
        pooled = [global_add_pool(h, data.batch)]
        for conv in self.convs:
            h = F.relu(conv(h, data.edge_index, data.edge_attr)) + h
            pooled.append(global_add_pool(h, data.batch))
        z = self.out(torch.cat(pooled, dim=1))
        return F.relu(z) if self.nonneg else z

IN_DIM, EDGE_DIM = len(ELEMENTS) + 1, len(BOND_TYPES)

class PairClassifier(nn.Module):
    """Model A: shared encoder, MLP on [zq, zt, zt - zq, zq * zt] -> logit."""
    def __init__(self):
        super().__init__()
        self.enc = GINEncoder(IN_DIM, EDGE_DIM, CFG["HIDDEN"], CFG["EMB_DIM"], CFG["N_LAYERS"])
        d = CFG["EMB_DIM"]
        self.head = nn.Sequential(nn.Linear(4 * d, 128), nn.ReLU(), nn.Linear(128, 1))

    def score(self, zq, zt):            # higher = more likely contained
        return self.head(torch.cat([zq, zt, zt - zq, zq * zt], dim=-1)).squeeze(-1)

class OrderEmbedder(nn.Module):
    """Model B (NeuroMatch): Q contained in T  <=>  zq <= zt element-wise.
    Embeddings are kept non-negative (ReLU), as in order embeddings (Vendrov et al.);
    in our tests this trained much better than unconstrained embeddings."""
    def __init__(self):
        super().__init__()
        self.enc = GINEncoder(IN_DIM, EDGE_DIM, CFG["HIDDEN"], CFG["EMB_DIM"], CFG["N_LAYERS"], nonneg=True)

    @staticmethod
    def violation(zq, zt):              # 0 if zq <= zt everywhere
        return F.relu(zq - zt).pow(2).sum(-1)

    def score(self, zq, zt):            # higher = more likely contained
        return -self.violation(zq, zt)

# %% [markdown]
# ## 7. Training
# Model A: binary cross-entropy. Model B: NeuroMatch max-margin loss,
# `E(q,t) = ||max(0, zq - zt)||^2`, minimised for positives and pushed above a margin for negatives.

# %%
def make_batches(pairs, queries, mol_graphs, batch_size, shuffle, rng):
    idx = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        chunk = [pairs[i] for i in idx[s:s + batch_size]]
        qb = Batch.from_data_list([queries[q]["graph"] for q, _, _ in chunk]).to(DEVICE)
        tb = Batch.from_data_list([mol_graphs[t] for _, t, _ in chunk]).to(DEVICE)
        y = torch.tensor([l for _, _, l in chunk], dtype=torch.float, device=DEVICE)
        yield qb, tb, y

def loss_fn(model, zq, zt, y):
    if isinstance(model, OrderEmbedder):
        e = model.violation(zq, zt)
        return (y * e + (1 - y) * F.relu(CFG["MARGIN"] - e)).mean()
    return F.binary_cross_entropy_with_logits(model.score(zq, zt), y)

@torch.no_grad()
def scores_on_pairs(model, pairs, queries, mol_graphs):
    model.eval(); out = []
    for qb, tb, _ in make_batches(pairs, queries, mol_graphs, 1024, False, None):
        out.append(model.score(model.enc(qb), model.enc(tb)).cpu())
    return torch.cat(out).numpy()

def roc_auc(y, s):
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def train(model, name):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    rng = random.Random(SEED)
    y_val = np.array([p[2] for p in val_pairs])
    hist = []
    for ep in range(1, CFG["EPOCHS"] + 1):
        model.train(); t0 = time.perf_counter(); tot, nb = 0.0, 0
        for qb, tb, y in make_batches(train_pairs, train_queries, train_mol_graphs, CFG["BATCH"], True, rng):
            loss = loss_fn(model, model.enc(qb), model.enc(tb), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        auc = roc_auc(y_val, scores_on_pairs(model, val_pairs, val_queries, val_mol_graphs))
        hist.append(dict(epoch=ep, loss=tot / nb, val_auc=auc))
        print(f"[{name}] epoch {ep:2d} | loss {tot / nb:.4f} | val AUC {auc:.4f} | {time.perf_counter() - t0:.1f} s")
    return pd.DataFrame(hist)

model_A = PairClassifier(); hist_A = train(model_A, "GIN + MLP (BCE)")
model_B = OrderEmbedder();  hist_B = train(model_B, "GIN order embedding")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for h, lab in [(hist_A, "GIN + MLP (BCE)"), (hist_B, "Order embedding")]:
    ax[0].plot(h.epoch, h.loss, label=lab); ax[1].plot(h.epoch, h.val_auc, label=lab)
ax[0].set(xlabel="epoch", ylabel="training loss"); ax[1].set(xlabel="epoch", ylabel="validation ROC-AUC")
ax[1].legend(); plt.tight_layout(); plt.savefig("results/training_curves.png", dpi=150); plt.show()

# %% [markdown]
# ## 8. Choose decision thresholds on the validation set
# - Each GNN alone: threshold that maximises validation F1.
# - Cascade: a looser threshold that keeps validation recall >= 0.98, since RDKit removes the false positives afterwards.

# %%
# The training pairs are roughly balanced, but real search is not: most molecules do NOT contain
# a given query. So thresholds are tuned on a validation *search*: validation queries against every
# validation molecule, with exact RDKit labels.
N_TUNE = min(200, len(val_queries))
tune_q = val_queries[:N_TUNE]
Y_val = np.array([[m.HasSubstructMatch(q["qmol"]) for m in val_mols] for q in tune_q])
print(f"Validation search: {N_TUNE} queries x {len(val_mols)} molecules, hit rate {Y_val.mean():.3f}")

@torch.no_grad()
def score_matrix(model, queries, z_targets):
    model.eval()
    zq = model.enc(Batch.from_data_list([q["graph"] for q in queries]).to(DEVICE))
    return torch.stack([model.score(zq[i].expand(z_targets.shape[0], -1), z_targets).cpu()
                        for i in range(len(queries))]).numpy()

def pick_thresholds(model):
    model.eval()
    with torch.no_grad():
        zval = model.enc(Batch.from_data_list(val_mol_graphs).to(DEVICE))
    S = score_matrix(model, tune_q, zval)
    best_f1, thr_f1 = -1, None
    for t in np.unique(np.quantile(S, np.linspace(0.3, 0.9999, 500))):
        pred = S >= t
        tp = (pred & Y_val).sum(); fp = (pred & ~Y_val).sum(); fn = (~pred & Y_val).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, thr_f1 = f1, t
    thr_rec = np.quantile(S[Y_val], 0.02)        # 98% of true hits score above this
    return float(thr_f1), float(thr_rec), float(best_f1)

thr_A, _, f1A = pick_thresholds(model_A)
thr_B, thr_B_casc, f1B = pick_thresholds(model_B)
print(f"Model A threshold {thr_A:.4f} (validation search F1 {f1A:.3f})")
print(f"Model B threshold {thr_B:.4f} (validation search F1 {f1B:.3f}), cascade threshold {thr_B_casc:.4f}")

# %% [markdown]
# ## 9. Retrieval benchmark on the held-out search database
# Test queries come from molecules that are in neither the training set nor the search DB,
# plus the hand-picked functional groups. For each query, every method returns the set of DB
# molecules it thinks contain Q; RDKit brute force gives the true set.
#
# One-time costs (DB fingerprints, DB embeddings) are reported separately: in a real system they are
# computed once and reused for every query.

# %%
def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def embed_graphs(model, graphs, bs=2048):
    model.eval(); out = []
    for s in range(0, len(graphs), bs):
        out.append(model.enc(Batch.from_data_list(graphs[s:s + bs]).to(DEVICE)))
    return torch.cat(out)

db_graphs = [mol_to_graph(m) for m in db_mols]
emb_time = {}
for name, model in [("A", model_A), ("B", model_B)]:
    sync(); t0 = time.perf_counter(); z = embed_graphs(model, db_graphs); sync()
    emb_time[name] = time.perf_counter() - t0
    if name == "A": zdb_A = z
    else: zdb_B = z
print(f"One-time DB embedding: model A {emb_time['A']:.1f} s, model B {emb_time['B']:.1f} s ({len(db_graphs)} molecules)")

@torch.no_grad()
def gnn_hits(model, zdb, q, thr):
    zq = model.enc(Batch.from_data_list([q["graph"]]).to(DEVICE))
    s = model.score(zq.expand(zdb.shape[0], -1), zdb)
    return (s >= thr).nonzero().squeeze(-1).cpu().numpy()

test_queries = sample_queries(qsrc_mols, CFG["N_TEST_QUERIES"], random.Random(SEED + 5))
for q in test_queries: q["name"] = "sampled"
all_test = test_queries + fg_queries
N = len(db_mols)

rows = []
for qi, q in enumerate(all_test):
    qm = q["qmol"]; r = dict(query=q["smarts"], kind=q["name"], q_atoms=q["graph"].num_nodes)

    t0 = time.perf_counter()
    truth = np.array([m.HasSubstructMatch(qm) for m in db_mols]); r["t_rdkit_brute"] = time.perf_counter() - t0
    truth_set = set(np.where(truth)[0]); r["n_true"] = len(truth_set)

    t0 = time.perf_counter()
    cand = np.where(fp_screen(pattern_fp_array(qm), db_fp))[0]; r["t_fp_screen"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    verified = [i for i in cand if db_mols[i].HasSubstructMatch(qm)]
    r["t_fp_rdkit"] = r["t_fp_screen"] + time.perf_counter() - t0
    r["fp_screen"] = set(cand.tolist()); r["fp_rdkit"] = set(verified)

    for key, model, zdb, thr in [("gnn_A", model_A, zdb_A, thr_A), ("gnn_B", model_B, zdb_B, thr_B),
                                 ("short_B", model_B, zdb_B, thr_B_casc)]:
        sync(); t0 = time.perf_counter(); hits = gnn_hits(model, zdb, q, thr); sync()
        r["t_" + key] = time.perf_counter() - t0; r[key] = set(hits.tolist())

    t0 = time.perf_counter()
    r["cascade_B"] = {i for i in r["short_B"] if db_mols[i].HasSubstructMatch(qm)}
    r["t_cascade_B"] = r["t_short_B"] + time.perf_counter() - t0
    r["truth"] = truth_set
    rows.append(r)
    if (qi + 1) % 50 == 0:
        print(f"  {qi + 1}/{len(all_test)} queries done")

# %%
METHODS = [("RDKit exact, brute force", "truth", "t_rdkit_brute"),
           ("Pattern FP screen only", "fp_screen", "t_fp_screen"),
           ("Pattern FP screen + RDKit", "fp_rdkit", "t_fp_rdkit"),
           ("GNN A: GIN + MLP", "gnn_A", "t_gnn_A"),
           ("GNN B: order embedding", "gnn_B", "t_gnn_B"),
           ("Cascade: GNN B shortlist + RDKit", "cascade_B", "t_cascade_B")]

def summarise(rows_subset):
    out = []
    for label, key, tkey in METHODS:
        tp = sum(len(r[key] & r["truth"]) for r in rows_subset)
        fp = sum(len(r[key] - r["truth"]) for r in rows_subset)
        fn = sum(len(r["truth"] - r[key]) for r in rows_subset)
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        out.append(dict(method=label, precision=round(p, 4), recall=round(rc, 4),
                        f1=round(2 * p * rc / max(p + rc, 1e-9), 4),
                        ms_per_query=round(1000 * np.mean([r[tkey] for r in rows_subset]), 2),
                        avg_returned=round(np.mean([len(r[key]) for r in rows_subset]), 1)))
    df = pd.DataFrame(out)
    base = df.loc[0, "ms_per_query"]
    df["speedup_vs_brute"] = (base / df["ms_per_query"]).round(1)
    return df

res_all = summarise(rows)
res_sampled = summarise([r for r in rows if r["kind"] == "sampled"])
res_fg = summarise([r for r in rows if r["kind"] != "sampled"])
print(f"Search DB: {N} molecules | {len(all_test)} queries | device: {DEVICE}\n")
print("ALL QUERIES"); print(res_all.to_string(index=False))
print("\nSAMPLED SUBGRAPH QUERIES"); print(res_sampled.to_string(index=False))
print("\nFUNCTIONAL-GROUP QUERIES"); print(res_fg.to_string(index=False))
res_all.to_csv("results/benchmark_all.csv", index=False)
res_sampled.to_csv("results/benchmark_sampled.csv", index=False)
res_fg.to_csv("results/benchmark_functional_groups.csv", index=False)
print(f"\nOne-time costs: DB fingerprints {t_db_fp:.1f} s, DB embeddings {emb_time['B']:.1f} s")

# %% [markdown]
# ## 10. Accuracy vs. speed trade-off

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for _, row in res_all.iterrows():
    ax[0].scatter(row.ms_per_query, row.f1, s=60)
    ax[0].annotate(row.method, (row.ms_per_query, row.f1), fontsize=8, xytext=(5, 3), textcoords="offset points")
ax[0].set(xscale="log", xlabel="time per query (ms, log scale)", ylabel="F1 vs exact RDKit",
          title=f"Accuracy vs speed ({N} molecules)")

# Precision-recall curve of the order-embedding model over thresholds (sampled queries)
sub = [r for r in rows if r["kind"] == "sampled"][:100]
zq_all = embed_graphs(model_B, [all_test[rows.index(r)]["graph"] for r in sub])
P, R = [], []
with torch.no_grad():
    S = torch.stack([model_B.score(zq_all[i].expand(N, -1), zdb_B).cpu() for i in range(len(sub))]).numpy()
Y = np.zeros_like(S, dtype=bool)
for i, r in enumerate(sub):
    Y[i, list(r["truth"])] = True
for t in np.quantile(S, np.linspace(0.5, 0.9999, 60)):
    pred = S >= t; tp = (pred & Y).sum()
    P.append(tp / max(pred.sum(), 1)); R.append(tp / max(Y.sum(), 1))
ax[1].plot(R, P, marker=".")
ax[1].set(xlabel="recall", ylabel="precision", title="Order-embedding GNN: threshold sweep")
plt.tight_layout(); plt.savefig("results/tradeoff.png", dpi=150); plt.show()

# %% [markdown]
# ## 11. Where does the GNN fail? (error analysis by query size)

# %%
size_rows = []
for r in [r for r in rows if r["kind"] == "sampled"]:
    tp = len(r["gnn_B"] & r["truth"])
    size_rows.append(dict(q_atoms=r["q_atoms"], recall=tp / max(len(r["truth"]), 1),
                          precision=tp / max(len(r["gnn_B"]), 1), n_true=r["n_true"]))
by_size = pd.DataFrame(size_rows).groupby("q_atoms").mean().round(3)
print(by_size)
by_size.to_csv("results/order_model_by_query_size.csv")

print("\nFunctional-group queries, order-embedding GNN:")
fg_tab = pd.DataFrame([dict(group=r["kind"], true_hits=r["n_true"], gnn_returned=len(r["gnn_B"]),
                            gnn_correct=len(r["gnn_B"] & r["truth"])) for r in rows if r["kind"] != "sampled"])
print(fg_tab.to_string(index=False))

# %%
torch.save({"model_A": model_A.state_dict(), "model_B": model_B.state_dict(), "cfg": CFG,
            "thr_A": thr_A, "thr_B": thr_B, "thr_B_casc": thr_B_casc}, "results/models.pt")
with open("results/config.json", "w") as f:
    json.dump(dict(CFG, device=str(DEVICE), db_fp_seconds=t_db_fp, db_embed_seconds=emb_time), f, indent=2)
print("Saved everything in ./results")
