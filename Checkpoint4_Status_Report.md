# CS 6012 — Autumn 2026: Open Project
## Intermediate Checkpoint #4: Status Report for Submission 2 (Code-Based Project)

**Date:** October 24, 2026  
**Team Members:**  
- Aayush Gupta (Roll No: 25m0140)  
- Srijal Gupta (Roll No: 25m0387)  

---

### 1. Title of the Project
**GNN Molecular Substructure Search: Accuracy vs. Speed Trade-offs Against Classical Fingerprint Screening**

### 2. Updates on Problem Statement
*Refinement to a Two-Stage Cascaded Screening Architecture:*  
In realistic chemical discovery databases ($50{,}000+$ molecules), end-to-end neural classification alone is insufficient because false negatives can cause chemists to miss critical active scaffolds. Consequently, we have formalized our problem into a **two-stage cascaded filtering paradigm**:
1. **Stage 1 (GNN Filter):** A learned order-embedding GNN rapidly screens the entire database to shortlist candidate molecules at a strict high-recall constraint ($\ge 98\%$).
2. **Stage 2 (Exact Verification):** RDKit’s exact subgraph matcher (`HasSubstructMatch`) is evaluated *only* on the shortlisted candidates, yielding $100\%$ precision.

The core research question is whether this learned neural cascade achieves a superior speed-accuracy Pareto frontier compared to the industry standard: RDKit’s 2048-bit `Chem.PatternFingerprint` screen followed by exact verification.

### 3. Difficulties Encountered
1. **Candidate Shortlist Overhead & Verification Bottleneck:**  
   While GNN B's order-embedding screen is exceptionally fast on GPU ($2.79\text{ ms}$ over 5,000 molecules), its lower precision at the $\ge 98\%$ recall operating point returns an average of $1{,}414$ candidate molecules (compared to only $703$ candidates for Pattern Fingerprints). Because exact RDKit verification takes $\sim 0.05\text{ ms}$ per molecule, verifying those extra candidates increases overall cascaded latency to $20.21\text{ ms}$ (vs. $5.61\text{ ms}$ for Pattern FP + RDKit). Tighter precision calibration is required to prevent verification overhead from dominating.
2. **Query Size Generalization Decay (4 vs. 12 Atoms):**  
   Empirical query-size breakdown revealed that while the GIN encoder performs well on small subgraphs (4–5 atoms: recall $83.4\%–83.7\%$), performance drops substantially on larger queries (9–12 atoms: recall falls to $28.4\%–43.6\%$). The 4-layer GINE receptive field and global sum-pooling struggle to preserve strict subgraph isomorphism constraints as topological diameter and cycle complexity grow.
3. **Hard Negative Mining Scalability:**  
   Training order embeddings requires hard negatives (pairs that pass bitwise fingerprint screening but lack exact topological containment). Generating these requires repeated RDKit exact calls during batch creation, creating a CPU-side bottleneck when scaling beyond 40,000 training pairs.

### 4. Contributions of Members
- **Aayush Gupta (25m0140):** GPU acceleration and CUDA runtime integration on RTX 3050, PyG `GINEConv` message-passing optimization, max-margin order loss violation tuning, threshold calibration curves for the cascade shortlist, and Checkpoint documentation.
- **Srijal Gupta (25m0387):** ZINC250k query generation stratified across atom sizes (4–12 atoms + 20 functional groups), hard-negative sampling pipeline, classical Pattern Fingerprint benchmarking suite, error analysis by query size, and updated schedule tracking.
- **Joint Deliverables:** Cascaded pipeline evaluation, error analysis documentation, presentation slide preparation, and GitHub repository maintenance.

### 5. GitHub Submission & Demonstrated Progress
- **GitHub Repository:** [https://github.com/srijal20/GNN-Substructure-Search-project](https://github.com/srijal20/GNN-Substructure-Search-project)  
  *(Active repository containing complete pipeline code `gnn_substructure_search.py`, interactive Colab notebook `GNN_Substructure_Search.ipynb` with 1-click badge, pilot CSV benchmark data in `results/pilot/`, tradeoff curves, and status reports.)*
- **Updated Schedule Spreadsheet:** Submitted as a standalone companion PDF (`Updated_Schedule_Spreadsheet.pdf`) and tracked online at: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1264hAmtv7T5bGPyuDbSqR52UGxm6RZCEa7NFxWk-cQ4/edit?usp=sharing).
