# GNN Molecular Substructure Search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srijal20/GNN-Substructure-Search-project/blob/main/GNN_Substructure_Search.ipynb)

CS 6012 Open Project (Submission 2). Srijal Gupta and Aayush Gupta.

**Question:** can a graph neural network find most molecules that contain a chemical pattern
faster than RDKit's exact substructure matcher?

## What is in the notebook
1. ZINC250k molecules, split into train / validation / search database / query sources (no overlap).
2. Queries: connected subgraphs (4 to 12 atoms) sampled from molecules, plus 20 common functional groups.
3. Ground truth: RDKit exact matching (`HasSubstructMatch`).
4. Baseline: RDKit pattern fingerprint screen (never misses a true hit).
5. GNN A: bond-aware GIN encoder (GINE) + MLP classifier, BCE loss.
6. GNN B: NeuroMatch-style order embeddings (query embedding <= target embedding).
7. Cascade: GNN B shortlists molecules, RDKit checks only the shortlist.
8. Benchmark: precision, recall, F1 and time per query for every method.

## How to run
Open `GNN_Substructure_Search.ipynb` in Google Colab, set Runtime > Change runtime type > T4 GPU,
then Runtime > Run all. Set `QUICK = True` for a 5-minute test. The dataset downloads automatically.
`gnn_substructure_search.py` is the same code as a plain script.

## Status (18 Oct 2026)
Pilot run on CPU (5k-molecule database) is in `results/pilot/`. The full-size GPU run is next.

| Method | Precision | Recall | ms / query |
|---|---|---|---|
| RDKit exact (brute force) | 1.00 | 1.00 | 31.8 |
| Pattern fingerprint only | 0.70 | 1.00 | 1.1 |
| Pattern fingerprint + RDKit | 1.00 | 1.00 | 5.6 |
| GNN A: GIN + MLP | 0.46 | 0.64 | 6.7 |
| GNN B: order embedding | 0.27 | 0.77 | 2.8 |
| Cascade: GNN B + RDKit | 1.00 | 0.98 | 20.2 |

## References
- R. Ying et al., Neural Subgraph Matching (NeuroMatch), 2020.
- K. Xu et al., How Powerful are Graph Neural Networks?, ICLR 2019.
- D. Rose et al., PharmacoMatch, ICLR 2025.
