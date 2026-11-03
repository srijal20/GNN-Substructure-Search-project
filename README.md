# GNN Molecular Substructure Search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srijal20/GNN-Substructure-Search-project/blob/main/GNN_Substructure_Search.ipynb)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![PyG 2.8](https://img.shields.io/badge/PyTorch_Geometric-2.8.1-orange.svg)](https://pyg.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.03-green.svg)](https://www.rdkit.org/)

**CS 6012: Deep Learning — Open Project (Submission 2: Code-Based Project)**  
**Department of Computer Science & Engineering, Indian Institute of Technology Bombay**  

**Team Members:**  
- **Aayush Gupta** (Roll No: 25m0140) — `25m0140@iitb.ac.in` / `aayushg416@gmail.com`  
- **Srijal Gupta** (Roll No: 25m0387) — `25m0387@iitb.ac.in` / `sirjalgupta@gmail.com`  

---

## 📌 Executive Summary

Exact chemical substructure search ($Q \subseteq T$) using combinatorial graph matching (e.g., Ullmann/VF2 in RDKit) scales exponentially in the worst case $\mathcal{O}(|V_T|^{|V_Q|})$ and becomes computationally prohibitive across large chemical libraries ($50{,}000+$ molecules). 

In this project, we design, implement, and benchmark Graph Neural Network (GNN) architectures to study the **speed-accuracy Pareto frontier** against industry-standard cheminformatics screening:
1. **Model A (Siamese GIN + MLP Classifier):** A bond-aware `GINEConv` encoder trained with Binary Cross-Entropy (BCE).
2. **Model B (NeuroMatch Order Embeddings):** A geometric partial-order embedder mapping molecules into a non-negative cone $\mathbb{R}_{\ge 0}^{64}$ where subgraph containment corresponds to element-wise dominance ($z_Q \le z_T$), trained via max-margin violation loss ($E(Q, T) = \|\max(0, z_Q - z_T)\|_2^2$).
3. **Two-Stage Cascaded Screening Pipeline:** The order-embedding GNN rapidly shortlists candidate molecules on GPU at a strict $\ge 98\%$ recall threshold, after which RDKit exact verification is executed *only* on the shortlist, guaranteeing $100\%$ precision with zero false positives.

---

## 📊 Comprehensive Benchmark Results

Evaluated across the held-out search database ($5{,}000$ to $50{,}000$ ZINC250k drug-like molecules) over 120+ test queries (sampled subgraphs of 4–12 atoms and 20 validated medicinal chemistry functional groups):

| Retrieval Method | Precision | Recall | F1-Score | Latency / Query | Speedup vs. Brute | Avg. Shortlist Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RDKit Exact (Brute Force)** | **1.0000** | **1.0000** | **1.0000** | 31.84 ms | $1.0\times$ | 494.2 |
| **Pattern FP screen only** | 0.7026 | **1.0000** | 0.8254 | **1.09 ms** | **$29.2\times$** | 703.3 |
| **Pattern FP + RDKit (Industrial Baseline)** | **1.0000** | **1.0000** | **1.0000** | **5.61 ms** | **$5.7\times$** | 494.2 |
| **GNN A: GIN + MLP Classifier** | 0.4613 | 0.6377 | 0.5353 | 6.66 ms | $4.8\times$ | 683.2 |
| **GNN B: Order Embedding alone** | 0.2695 | 0.7709 | 0.3993 | **2.79 ms** | **$11.4\times$** | 1,413.9 |
| **Cascade: GNN B Shortlist + RDKit** | **1.0000** | 0.9824 | 0.9911 | 20.21 ms | $1.6\times$ | 485.5 |

### 🔬 Core Scientific Insights:
- **The False-Positive Verification Bottleneck:** GNN B alone performs vector screening in just **$2.79\text{ ms}$** ($11.4\times$ speedup). However, to guarantee $\ge 98\%$ recall, its decision threshold returns $1{,}413.9$ candidates (vs. $703.3$ for Pattern Fingerprints). Verifying those false positives in RDKit ($\sim 0.012\text{ ms/mol}$) consumes $17.4\text{ ms}$, dropping cascaded speedup to $1.6\times$. *A learned filter must achieve extreme precision at high recall to beat classical bitwise screens.*
- **Receptive Field Decay by Query Size:** On small subgraphs (4–5 atoms), GNN B achieves **$>83\%$ recall**. On larger subgraphs (10–12 atoms), recall drops to **$28\%–44\%$** because the 4-layer GINE receptive field and whole-graph sum-pooling struggle to preserve long-range cycle isomorphism as graph diameter expands.

---

## 🏗️ System Architecture

```
                       +----------------------------------+
                       |   Query Fragment Q (SMARTS/Graph)|
                       +-----------------+----------------+
                                         |
                       +-----------------v----------------+
                       |    GINEConv Bond-Aware Encoder   |
                       +-----------------+----------------+
                                         | z_Q in R+^64
                                         v
   +-----------------------+     Vectorized Order Check     +-----------------------+
   | Target DB Embeddings  | ===>  E(Q, T) = ||max(0,     | ===> Candidate Shortlist
   | z_T in R+^64 (Precmp) |        z_Q - z_T)||^2 <= thr  |      (Recall >= 98%)
   +-----------------------+                                +-----------+-----------+
                                                                        |
                                                                        v
                                                            +-----------------------+
                                                            | RDKit HasSubstruct    |
                                                            | Match (Exact Check)   |
                                                            +-----------+-----------+
                                                                        |
                                                                        v
                                                            +-----------------------+
                                                            | Final Exact Matches   |
                                                            | (100% Prec, 98% Rec)  |
                                                            +-----------------------+
```

---

## 📁 Repository Structure

```
gnn-substructure-search/
├── Final_Project_Report.pdf         # Official 2-Page publication-quality final project report
├── Final_Project_Report.md          # Comprehensive final research report in Markdown
├── Video_Presentation_Script.md     # 2-Minute video recording script & slide walkthrough
├── GNN_Substructure_Search.ipynb    # End-to-end interactive Jupyter/Colab notebook
├── gnn_substructure_search.py       # Standalone modular Python pipeline script
├── requirements.txt                 # Project environment dependencies
├── README.md                        # Documentation, architecture, and results summary
├── ScheduleForOpenProject.xlsx      # Original project milestone tracking spreadsheet
├── Updated_Schedule_Spreadsheet.pdf # Formal 1-page milestone schedule tracking PDF
├── Checkpoint2_Status_Report.pdf    # Checkpoint #2 status report PDF
├── Checkpoint4_Status_Report.pdf    # Checkpoint #4 status report PDF
└── results/
    └── pilot/
        ├── benchmark_all.csv               # Quantitative retrieval metrics across all queries
        ├── benchmark_functional_groups.csv # Performance on 20 medicinal chemistry functional groups
        ├── benchmark_sampled.csv           # Performance on sampled BFS subgraph queries
        ├── order_model_by_query_size.csv   # Precision/recall breakdown across 4-12 query atoms
        ├── config.json                     # Training hyperparameters & precomputation latencies
        ├── tradeoff.png                    # Precision-recall curves & latency-accuracy Pareto plot
        └── training_curves.png             # Training loss and validation ROC-AUC curves
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended for quick evaluation)
1. Click the **[Open In Colab](https://colab.research.google.com/github/srijal20/GNN-Substructure-Search-project/blob/main/GNN_Substructure_Search.ipynb)** badge.
2. Go to **Runtime > Change runtime type > T4 GPU**.
3. Run all cells (`Ctrl + F9`). The notebook automatically installs dependencies, downloads ZINC250k, trains both GNN models, and produces the benchmark tables and plots.
4. Set `QUICK = True` in Cell 1 for a 5-minute smoke test, or `QUICK = False` for the full GPU run.

### Option 2: Local Run (with GPU CUDA acceleration)
```powershell
# 1. Clone the repository
git clone https://github.com/srijal20/GNN-Substructure-Search-project.git
cd GNN-Substructure-Search-project

# 2. Install dependencies (PyTorch with CUDA 12.4 + PyG + RDKit)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric rdkit pandas matplotlib reportlab pypdf

# 3. Execute the full pipeline
python gnn_substructure_search.py
```

---

## 👥 Division of Work & Contributions

- **Aayush Gupta (25m0140):** GPU acceleration & CUDA runtime setup on RTX 3050; graph featurization pipeline (`atom_feat`, `bond_feat`, `mol_to_graph`); GINEConv backbone, Model A (BCE classification), and Model B (NeuroMatch order embedding violation loss); ROC-AUC evaluation; high-recall threshold optimization algorithm; final report authoring and video script preparation.
- **Srijal Gupta (25m0387):** ZINC250k data loading and disjoint dataset partitioning; breadth-first connected query subgraph generation (4 to 12 atoms); functional group query curation (20 chemical classes); hard negative pair mining against fingerprint screens; RDKit Pattern Fingerprint and exact baseline evaluation harness; query-size error ablation; schedule tracking.
- **Joint Work:** Cascaded search architecture design, experimental benchmarking, report editing, and video recording.

---

## 📚 References

1. K. Xu, W. Hu, J. Leskovec, and S. Jegelka, *"How Powerful are Graph Neural Networks?"*, International Conference on Learning Representations (ICLR), 2019.
2. R. Ying, Z. Lou, J. You, C. Zheng, et al., *"Neural Subgraph Matching"*, Neural Information Processing Systems (NeurIPS), 2020.
3. D. Rose et al., *"PharmacoMatch: Efficient 3D Pharmacophore Screening via Neural Subgraph Matching"*, International Conference on Learning Representations (ICLR), 2025.
4. I. Vendrov, R. Kiros, S. Fidler, and R. Urtasun, *"Order-Embeddings of Images and Language"*, International Conference on Learning Representations (ICLR), 2016.
5. G. Landrum et al., *"RDKit: Open-source cheminformatics"*, [https://github.com/rdkit/rdkit](https://github.com/rdkit/rdkit).
