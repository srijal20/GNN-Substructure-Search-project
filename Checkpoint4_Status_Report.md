# CS 6012 — Autumn 2026: Open Project
## Intermediate Checkpoint #4: Status Report for Submission 2 (Code-Based Project)

**Date:** October 24, 2026  
**Team Members:**  
- Aayush Gupta (Roll No: 25m0140)  
- Srijal Gupta (Roll No: 25m0387)  

---

### 1. Title of the Project
**GNN Molecular Substructure Search: Accuracy vs. Speed Trade-offs Against Classical Fingerprint Screening**

### 2. Updates on Problem Statement & Cascaded Framework
*Refinement to a Two-Stage Cascaded Screening Architecture:*  
Exact molecular substructure matching ($Q \subseteq T$) using combinatorial algorithms (VF2/Ullmann in RDKit) scales exponentially in the worst case $\mathcal{O}(|V_T|^{|V_Q|})$ and becomes computationally prohibitive across large chemical libraries ($50{,}000+$ molecules). Because pure neural classifiers suffer from false negatives that can discard critical bioactive scaffolds, we have formally refined our pipeline into a **two-stage cascaded filtering paradigm**:
1. **Stage 1 (Learned GNN Filter):** A bond-aware Graph Isomorphism Network with NeuroMatch-style order embeddings ($z_Q \le z_T$ element-wise in $\mathbb{R}_{\ge 0}^{64}$) screens the database in $<3\text{ ms}$ on GPU to shortlist candidate molecules at a strict high-recall operating constraint ($\ge 98\%$).
2. **Stage 2 (Exact Verification):** RDKit’s exact `HasSubstructMatch` is evaluated *only* on the shortlisted candidate pool, guaranteeing $100\%$ precision with zero false positives.

The research objective is quantifying the Pareto frontier (latency vs. recall) of this learned neural cascade against the industrial gold standard: RDKit’s 2048-bit `Chem.PatternFingerprint` bitwise screening (`(fp_T & fp_Q) == fp_Q`) followed by exact verification.

### 3. Empirical Results & Difficulties Encountered
Our pilot benchmark across 5,000 ZINC molecules (10,000 training queries, 4-layer GINE, embedding dim 64) revealed key quantitative trade-offs:
- **Baseline Exact RDKit (Brute Force):** $31.84\text{ ms/query}$ | Precision: $1.00$ | Recall: $1.00$ | Speedup: $1.0\times$
- **Pattern Fingerprint + RDKit (Industrial Baseline):** **$5.61\text{ ms/query}$** | Precision: $1.00$ | Recall: $1.00$ | Speedup: **$5.7\times$** (Avg. candidates: $703.3$)
- **GNN B (Order Embedding Filter alone):** **$2.79\text{ ms/query}$** | Precision: $0.27$ | Recall: $0.77$ | Speedup: $11.4\times$ (Avg. candidates: $1{,}413.9$)
- **Cascade (GNN B Shortlist + RDKit):** **$20.21\text{ ms/query}$** | Precision: $1.00$ | Recall: $0.982$ | Speedup: $1.6\times$ (Shortlist: $485.5$)

**Core Difficulties & Technical Insights:**
1. **Shortlist Overhead & Verification Bottleneck:** GNN B is over $2\times$ faster at vector screening than the fingerprint screen ($2.79\text{ ms}$ vs. $1.09\text{ ms}$ without precomputed bit-arrays). However, at the $\ge 98\%$ recall threshold, its false positive rate produces $\sim 1{,}414$ candidates (vs. $703$ for Pattern FP). At $\sim 0.05\text{ ms}$ per RDKit check, verifying those false positives creates an execution bottleneck, lowering cascaded speedup from a theoretical $11\times$ to $1.6\times$.
2. **Query-Size Receptive Field Decay (4 vs. 12 Atoms):** Empirical breakdown by query size demonstrated high recall on small fragments (4 atoms: $83.4\%$; 5 atoms: $83.7\%$), but sharp degradation on larger subgraphs (9 atoms: $67.8\%$; 10 atoms: $28.4\%$; 12 atoms: $43.6\%$). The 4-layer GINEConv receptive field and global sum-pooling struggle to distinguish long-range cycle topologies as molecular diameter expands.
3. **CPU Preprocessing Bottleneck in Hard-Negative Mining:** Constructing balanced training pairs requiring hard negatives (pairs passing bitwise fingerprint screens but failing exact topological matching) requires $>40{,}000$ CPU-bound RDKit exact calls during batching.

### 4. Contributions of Members
- **Aayush Gupta (25m0140):** GPU acceleration and CUDA runtime integration on RTX 3050, PyG GINEConv message-passing optimization, max-margin order loss violation tuning, threshold calibration curves for the cascade shortlist, and Checkpoint documentation.
- **Srijal Gupta (25m0387):** ZINC250k query generation stratified across atom sizes (4–12 atoms + 20 functional groups), hard-negative sampling pipeline, classical Pattern Fingerprint benchmarking suite, error analysis by query size, and schedule tracking.
- **Joint Deliverables:** Cascaded pipeline evaluation, error analysis documentation, presentation slide preparation, and GitHub repository maintenance.

### 5. Codebase Packaging & Submission Progress
- **Codebase & Artifact Readiness:** The full end-to-end Python pipeline (`gnn_substructure_search.py`), interactive Colab notebook (`GNN_Substructure_Search.ipynb`), benchmark results suite (`results/pilot/`), ROC/tradeoff plots, and model configurations have been implemented, tested, and validated.
- **Repository Submission Status:** As planned, the codebase has been structured and documented locally with automated environment scripts and verified dependency specifications. The final repository integration and upload to the designated course repository are actively in progress and will be delivered alongside the final submission package.
- **Updated Schedule Spreadsheet:** Submitted as companion PDF (`Updated_Schedule_Spreadsheet.pdf`) and tracked online at: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1264hAmtv7T5bGPyuDbSqR52UGxm6RZCEa7NFxWk-cQ4/edit?usp=sharing).
