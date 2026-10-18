# CS 6012 — Autumn 2026: Open Project
## Intermediate Checkpoint #2: Status Report for Submission 2 (Code-Based Project)

**Date:** October 19, 2026  
**Team Members:**  
- Aayush Gupta (Roll No: 25m0140)  
- Srijal Gupta (Roll No: 25m0387)  

---

### 1. Title of the Project
**GNN Molecular Substructure Search: Accuracy vs. Speed Trade-offs Against Classical Fingerprint Screening**

### 2. Problem Statement
Exact molecular substructure search using combinatorial graph matching (e.g., RDKit’s Ullmann/VF2-based `HasSubstructMatch`) is reliable but computationally intensive, scaling poorly over large chemical databases. We train a Graph Neural Network (GNN) to act as an ultrafast approximate filter predicting whether a chemical query fragment $Q$ is contained within a target molecule $T$. The goal is not to replace exact matching, but to study the accuracy-speed Pareto frontier when pairing a learned GNN shortlist with exact RDKit verification, benchmarked directly against industrial cheminformatics standards (`Chem.PatternFingerprint`).

### 3. Extensions to Prior Work: [IN PROGRESS]
1. **NeuroMatch-Style Order Embeddings:** We extended beyond standard pairwise graph classification by formulating substructure containment as a partial order in a learned non-negative embedding space ($z_Q \le z_T$ element-wise), optimized via max-margin violation loss ($E(Q, T) = \|\max(0, z_Q - z_T)\|_2^2$). This enables precomputing target database embeddings once, reducing search to vectorized element-wise comparisons.
2. **Bond-Aware Molecular Graph Encoding:** Adapted generic subgraph models to molecules by utilizing `GINEConv` layers with 10-dimensional one-hot atom features and 4-dimensional bond features matching SMARTS query semantics.
3. **Two-Stage Cascaded Screening:** Developed a cascaded retrieval architecture where the order-embedding model shortlists candidates at a calibrated $98\%$ recall threshold, followed by RDKit verification on only the shortlist.

### 4. How Do We Feel?
**In-the-zone.** We have implemented the complete end-to-end pipeline in PyTorch Geometric and RDKit, and completed a pilot validation benchmark on a 5,000-molecule database.

### 5. Contributions of Members
- **Aayush Gupta (25m0140):** Implemented PyG graph featurization (atom and bond one-hot encoding), GINEConv Siamese encoder, Model A (BCE) and Model B (NeuroMatch order embedding) loss implementations, decision threshold optimization, and Colab integration.
- **Srijal Gupta (25m0387):** Built ZINC250k data loading and partitioning, subgraph query generation (4–12 atoms + 20 functional groups), RDKit ground-truth labeling, Pattern Fingerprint screening baseline, and comparative timing/profiling benchmarks.
- **Joint Work:** Cascade screening formulation, pilot error analysis by query size, report and presentation documentation.

### 6. Repository & Updated Schedule
- **GitHub Repository:** [https://github.com/srijal20/GNN-Substructure-Search-project](https://github.com/srijal20/GNN-Substructure-Search-project)  
  *(Contains full source code, pilot results CSVs, tradeoff plots, and interactive Colab notebook)*
- **Schedule Spreadsheet:** [https://docs.google.com/spreadsheets/d/1264hAmtv7T5bGPyuDbSqR52UGxm6RZCEa7NFxWk-cQ4/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1264hAmtv7T5bGPyuDbSqR52UGxm6RZCEa7NFxWk-cQ4/edit?usp=sharing)
