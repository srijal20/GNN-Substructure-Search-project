# GNN Molecular Substructure Search: Accuracy vs. Speed Trade-offs Against Classical Fingerprint Screening

**CS 6012: Deep Learning — Open Project (Submission 2: Final Report)**  
**Department of Computer Science & Engineering, IIT Bombay**  
**Date:** November 03, 2026  

**Team Members:**  
- **Aayush Gupta** (Roll No: 25m0140) — `25m0140@iitb.ac.in` / `aayushg416@gmail.com`  
- **Srijal Gupta** (Roll No: 25m0387) — `25m0387@iitb.ac.in` / `sirjalgupta@gmail.com`  

**Repository:** [https://github.com/srijal20/GNN-Substructure-Search-project](https://github.com/srijal20/GNN-Substructure-Search-project)  
**Interactive Colab Notebook:** [Open in Colab](https://colab.research.google.com/github/srijal20/GNN-Substructure-Search-project/blob/main/GNN_Substructure_Search.ipynb)  

---

## Abstract

Exact molecular substructure search is an essential primitive in computational drug discovery, chemical catalog screening, and lead optimization. Given a molecular query pattern $Q$, the objective is identifying all database molecules $T$ containing $Q$ as an induced subgraph ($Q \subseteq T$). Exact combinatorial algorithms (e.g., Ullmann, VF2) guarantee correctness but suffer from worst-case exponential time complexity $\mathcal{O}(|V_T|^{|V_Q|})$, becoming a major bottleneck when scaling to millions of molecules or high query throughput. In this project, we design, implement, and benchmark Graph Neural Network (GNN) architectures for approximate substructure retrieval on the ZINC250k drug-like chemical space. We evaluate: (1) a bond-aware Siamese Graph Isomorphism Network (`GINEConv`) with an MLP classifier trained via Binary Cross-Entropy (Model A), and (2) a geometric partial-order embedder based on NeuroMatch (Model B) that maps molecules into a non-negative cone $\mathbb{R}_{\ge 0}^{64}$ where substructure containment corresponds to element-wise dominance ($z_Q \le z_T$). 

Crucially, because drug discovery workflows cannot tolerate missing bioactive scaffolds (false negatives), we formalize a **two-stage cascaded filtering paradigm**: the learned order-embedding GNN rapidly filters the chemical database at a calibrated $\ge 98\%$ recall operating point, after which RDKit exact verification is executed *only* on the candidate shortlist, guaranteeing $100\%$ precision. We benchmark our neural models against brute-force RDKit and industry-standard RDKit 2048-bit `Chem.PatternFingerprint` bitwise screening. Our empirical results reveal that while GNN order embeddings achieve an ultrafast $11.4\times$ speedup during candidate retrieval ($2.79\text{ ms}$ vs. $31.84\text{ ms}$ brute force), their false-positive rate produces $1{,}413.9$ candidates (compared to $703.3$ for Pattern Fingerprints), causing downstream RDKit verification to dominate overall cascaded latency ($20.21\text{ ms}$ vs. $5.61\text{ ms}$ for Pattern FP + RDKit). Furthermore, query-size ablation reveals severe performance degradation for large subgraphs ($>83\%$ recall for 4–5 atoms vs. $28\%–44\%$ recall for 10–12 atoms), demonstrating the structural limitations of standard sum-pooled GIN representations for multi-hop cycle preservation.

---

## 1. Introduction & Research Motivation

In cheminformatics and molecular discovery, chemists frequently query compound databases containing millions of molecules (e.g., ZINC, ChEMBL, PubChem) to retrieve molecules containing specific pharmacophores, functional groups, or core heterocyclic scaffolds. Exact substructure matching is formally modeled as the **subgraph isomorphism problem**: given query graph $G_Q = (V_Q, E_Q)$ and target graph $G_T = (V_T, E_T)$, determine if there exists an injective mapping $f: V_Q \to V_T$ such that $(u, v) \in E_Q \implies (f(u), f(v)) \in E_T$ with matching atom and bond labels.

While modern cheminformatics toolkits such as RDKit implement optimized subgraph isomorphism routines based on the VF2 and Ullmann algorithms, these approaches face fundamental computational scaling limits:
1. **Combinatorial Complexity:** In the worst case, subgraph isomorphism is NP-complete. For queries with dense aromatic rings and symmetrical heteroatoms, backtracking searches scale exponentially.
2. **Database Scale:** In modern virtual screening, scanning $10^7$ molecules sequentially for thousands of queries is computationally prohibitive without indexing or pre-filtering.

To avoid running exact graph matching across every molecule, the chemical software industry relies on **molecular fingerprints** (such as RDKit’s `Chem.PatternFingerprint`). These fingerprints hash atom paths and ring systems into fixed-size bit vectors designed so that if $Q \subseteq T$, then $\text{fp}(Q) \subseteq \text{fp}(T)$ (i.e., every bit set in $Q$'s fingerprint must also be set in $T$'s fingerprint). This property allows rapid screening using bitwise operations (`(fp_T & fp_Q) == fp_Q`), guaranteeing zero false negatives.

### Research Questions
In this open project, we investigate whether learned deep learning models can improve upon or provide an alternative to classical fingerprints:
1. **Hypothesis 1:** Can a Graph Neural Network (GNN) learn to predict molecular substructure containment accurately from atom and bond topology alone?
2. **Hypothesis 2:** Can a geometric order-embedding formulation ($z_Q \le z_T$) allow one-time offline database pre-embedding, enabling faster approximate retrieval than bitwise fingerprint screening?
3. **Hypothesis 3:** In a practical two-stage cascaded system (GNN Filter $\to$ Exact Verification), does the neural filter achieve a superior accuracy-speed Pareto frontier compared to industrial fingerprint cascades?

> [!NOTE]
> ### 🧸 Intuitive Problem Formulation: The Lego Analogy & Airport Security Checkpoint
> To understand molecular substructure search intuitively, consider a toy box of **50,000 complex Lego models** (drug molecules like Aspirin, Paracetamol, Caffeine). A chemist hands us a **small Lego piece** of 6 connected blocks (a query fragment $Q$, like a Pyridine ring) and asks: *"Which of our 50,000 models contain this exact piece embedded inside them?"*
> - **Exact Matching (RDKit brute force):** Manually checks every atom and bond under a microscope. 100% accurate, but combinatorially explosive on massive libraries (31.8 ms/query).
> - **Classical Baseline (Pattern Fingerprints):** Generates a 2048-bit checklist (barcode) for each molecule. If the query requires a feature missing in the target, the target is rejected in 1 ms with guaranteed zero false negatives.
> - **Learned GNN Order Embedding:** AI projects graphs into a non-negative cone $\mathbb{R}_{\ge 0}^{64}$ where containment corresponds to vector dominance ($z_Q \le z_T$), enabling instantaneous GPU matrix screening (2.8 ms).
> - **Two-Stage Cascade (Airport Security Model):** Guard 1 (GNN) acts as a walk-through metal detector, shortlisting candidates at $\ge 98\%$ recall; Guard 2 (RDKit) performs a thorough pat-down only on suspects to guarantee 100% precision with zero false positives.

---

## 2. Background & Related Work

### 2.1 Graph Isomorphism Networks (GIN & GINE)
Standard message passing neural networks (MPNNs) aggregate neighbor features using mean, max, or sum operators. Xu et al. (ICLR 2019) demonstrated that mean and max aggregators fail to distinguish basic multiset structures, whereas **sum aggregation** possesses maximal discriminative power equivalent to the 1-Weisfeiler-Lehman (1-WL) graph isomorphism test. The Graph Isomorphism Network (GIN) updates node representations as:
$$h_v^{(k)} = \text{MLP}^{(k)} \left( \left(1 + \epsilon^{(k)}\right) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)$$
In molecular graphs, bond types (single, double, triple, aromatic) are crucial chemical constraints. We utilize **`GINEConv`** (Hu et al., 2020), which incorporates edge feature vectors $e_{uv}$ directly into the message formulation:
$$h_v^{(k)} = \text{MLP}^{(k)} \left( \left(1 + \epsilon^{(k)}\right) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \text{ReLU}\left(h_u^{(k-1)} + e_{uv}\right) \right)$$

### 2.2 Neural Subgraph Matching & Order Embeddings (NeuroMatch)
Ying et al. (NeurIPS 2020) introduced **NeuroMatch**, demonstrating that subgraph relationships naturally define a **partial order** (poset) over graph space. In order embeddings (Vendrov et al., ICLR 2016), entities are mapped into a non-negative vector space $\mathbb{R}_{\ge 0}^d$ where the partial ordering $Q \subseteq T$ is enforced by the element-wise inequality:
$$z_Q \le z_T \iff \forall i \in \{1, \dots, d\}, \; z_{Q, i} \le z_{T, i}$$
This formulation has an immense computational advantage: the embedding of every target molecule $z_T \in \mathbb{R}^d$ in a 50,000-molecule database can be computed **once offline**. When a new query $Q$ arrives, searching the entire database reduces to a single vectorized matrix subtraction on GPU:
$$E(Q, T) = \|\max(0, z_Q - z_T)\|_2^2$$
If $Q \subseteq T$, the penalty $E(Q, T) = 0$; if violated, $E(Q, T) > 0$.

### 2.3 Classical Substructure Screening Baselines
- **RDKit Exact (`HasSubstructMatch`):** Exact combinatorial backtracking. Correct by definition ($100\%$ precision, $100\%$ recall), but slowest.
- **RDKit Pattern Fingerprint (`Chem.PatternFingerprint`):** A 2048-bit specialized fingerprint that encodes atom types, degree, ring membership, and connectivity paths up to length 5. It is mathematically guaranteed that $Q \subseteq T \implies \text{fp}(Q) \subseteq \text{fp}(T)$. It never produces false negatives ($\text{Recall} = 1.0$), though it admits false positives due to bit hash collisions. (Note: Morgan/ECFP circular fingerprints do *not* possess this containment guarantee).

---

## 3. Methodology & System Architecture

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

### 3.1 Molecular Graph Featurization
We construct PyTorch Geometric `Data` graph objects preserving chemical semantics identical to RDKit SMARTS matching:
- **Node Features ($x \in \mathbb{R}^{|V| \times 10}$):** 10-dimensional one-hot vector representing the atom's atomic number: $\text{C (6)}, \text{N (7)}, \text{O (8)}, \text{F (9)}, \text{P (15)}, \text{S (16)}, \text{Cl (17)}, \text{Br (35)}, \text{I (53)}$, plus an "other" bucket.
- **Edge Features ($e \in \mathbb{R}^{|E| \times 4}$):** 4-dimensional one-hot vector encoding bond type: $\text{Single}, \text{Double}, \text{Triple}, \text{Aromatic}$.
- **Bidirectional Edges:** Directed edges are created in both directions ($u \to v$ and $v \to u$) with identical bond attributes.

### 3.2 GNN Encoder Architecture
Both models share a core backbone consisting of:
1. Linear input projection: $\mathbb{R}^{10} \to \mathbb{R}^{64}$.
2. 4 sequential `GINEConv` layers with 2-layer MLP message transforms (`Linear(64, 64) -> ReLU -> Linear(64, 64)`) and learnable $\epsilon$.
3. Multi-scale Jumping Knowledge / Layer Pooling: Node features from all intermediate layers (0 to 4) are pooled globally using `global_add_pool` and concatenated into a $5 \times 64 = 320$-dimensional multi-hop representation.
4. Final projection MLP: $\mathbb{R}^{320} \to \mathbb{R}^{64}$.

### 3.3 Model A: Siamese GIN + MLP Classifier
- The final graph vectors $z_Q, z_T \in \mathbb{R}^{64}$ are combined into a 256-dimensional interaction vector: $[z_Q, \; z_T, \; z_T - z_Q, \; z_Q \odot z_T]$.
- A classification head (`Linear(256, 128) -> ReLU -> Linear(128, 1)`) predicts a containment logit.
- Trained using Binary Cross-Entropy (BCE) with logits loss.

### 3.4 Model B: NeuroMatch Non-Negative Order Embeddings
- The projection head applies a final $\text{ReLU}$ activation, constraining embeddings strictly to the non-negative orthant $\mathbb{R}_{\ge 0}^{64}$.
- For a pair $(Q, T)$ with label $y \in \{0, 1\}$:
  $$E(Q, T) = \|\max(0, z_Q - z_T)\|_2^2$$
  $$\mathcal{L}(Q, T) = y \cdot E(Q, T) + (1 - y) \cdot \max(0, M - E(Q, T))$$
  where $M = 1.0$ is the violation margin. Positive pairs are pulled into containment ($E \to 0$), while negative pairs are penalized if their violation energy falls below margin $M$.

### 3.5 Two-Stage Cascaded Screening Pipeline
Real-world chemical search queries are sparse: $<5\%$ of database molecules contain a given specific scaffold.
1. **Offline Phase:** Compute and store $z_T \in \mathbb{R}^{64}$ for all $50{,}000$ database molecules.
2. **Online Query Phase:**
   - Embed query: $z_Q = \text{Encoder}(G_Q)$ ($<0.5\text{ ms}$).
   - Compute violation matrix $E(Q, T)$ against all stored $z_T$ using PyTorch tensor operations.
   - Select candidate shortlist $\mathcal{S} = \{T \in \mathcal{D} \mid E(Q, T) \le \tau_{\text{cascade}}\}$. The threshold $\tau_{\text{cascade}}$ is pre-calibrated on held-out validation searches to guarantee $\ge 98\%$ recall.
   - Run RDKit exact `mol.HasSubstructMatch(qmol)` exclusively on $\mathcal{S}$.

---

## 4. Experimental Setup & Benchmarks

### 4.1 Dataset & Disjoint Partitioning
We utilized the ZINC250k drug-like molecule library ($249{,}455$ clean drug-like compounds). To evaluate strict generalization without data leakage, we partitioned the dataset into four disjoint subsets:
- **Training Pool:** $40{,}000$ molecules
- **Validation Pool:** $4{,}000$ molecules
- **Search Database ($\mathcal{D}$):** $50{,}000$ molecules (completely unseen during training)
- **Held-out Query Sources:** $2{,}000$ molecules

### 4.2 Query Sampling & Hard-Negative Mining
1. **Sampled Subgraphs:** Connected subgraphs of size $k \in [4, 12]$ atoms sampled via breadth-first frontier growth from held-out molecules. Stereo chemistry tags were stripped to ensure valid SMARTS matching.
2. **Realistic Functional Groups:** 20 verified chemical functional groups representing real medicinal chemistry search queries (benzene, pyridine, thiophene, furan, carboxylic acid, amide, ester, urea, nitrile, sulfonamide, trifluoromethyl, piperidine, piperazine, morpholine, phenol, aniline, chlorobenzene, indole, biphenyl, ketone).
3. **Hard Negative Mining:** To force the GNN to learn connectivity rather than simple atom counts, for each query we mined "hard negatives": molecules that pass the 2048-bit Pattern Fingerprint bitwise screen but fail exact topological subgraph isomorphism.

### 4.3 Training Regimes & Hardware
- **Hardware:** Local NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM, CUDA 12.4, PyTorch 2.6.0+cu124) and Google Colab (T4 GPU).
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, weight decay $= 0$).
- **Batch Size:** 256 query-target pairs.
- **Epochs:** 20 epochs with ROC-AUC validation checkpointing.

---

## 5. Empirical Results & Comparative Analysis

The table below summarizes the comprehensive benchmark evaluated across the search database for all test queries (sampled subgraphs and functional groups).

| Retrieval Method | Precision | Recall | F1-Score | Latency / Query | Speedup vs. Brute | Avg. Candidates Returned |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RDKit Exact (Brute Force)** | **1.0000** | **1.0000** | **1.0000** | 31.84 ms | $1.0\times$ | 494.2 |
| **Pattern FP screen only** | 0.7026 | **1.0000** | 0.8254 | **1.09 ms** | **$29.2\times$** | 703.3 |
| **Pattern FP screen + RDKit (Baseline)** | **1.0000** | **1.0000** | **1.0000** | **5.61 ms** | **$5.7\times$** | 494.2 |
| **GNN A: GIN + MLP Classifier** | 0.4613 | 0.6377 | 0.5353 | 6.66 ms | $4.8\times$ | 683.2 |
| **GNN B: Order Embedding alone** | 0.2695 | 0.7709 | 0.3993 | **2.79 ms** | **$11.4\times$** | 1,413.9 |
| **Cascade: GNN B Shortlist + RDKit** | **1.0000** | 0.9824 | 0.9911 | 20.21 ms | $1.6\times$ | 485.5 |

### One-Time Precomputation Costs (Offline)
- **Database Fingerprints (Pattern FP, 5,000 DB):** $1.36\text{ seconds}$
- **Database Graph Embeddings (GNN B, 5,000 DB):** $0.67\text{ seconds}$

---

## 6. Deep Error Analysis & Scientific Discussion

### 6.1 The False Positive Verification Bottleneck
A central insight of this work is explaining why GNN B achieves an $11.4\times$ speedup when run standalone ($2.79\text{ ms}$), but drops to only $1.6\times$ speedup ($20.21\text{ ms}$) in the cascade:
- In order to guarantee $\ge 98\%$ recall, the decision threshold $\tau_{\text{cascade}}$ must be relaxed.
- At this loose threshold, GNN B returns an average of **$1{,}413.9$ candidates** per query—almost **double** the candidate pool returned by the classical Pattern Fingerprint screen ($703.3$ candidates).
- Because exact RDKit verification takes approximately $\sim 0.04 - 0.05\text{ ms}$ per molecule, verifying $1{,}414$ candidates in RDKit consumes:
  $$1414 \times 0.012\text{ ms} \approx 17.4\text{ ms}$$
- **Key Takeaway:** The bottleneck in a two-stage neural retrieval cascade is not the neural screening speed, but rather the **precision of the neural filter at high recall**. Even small false-positive leaks translate into massive downstream exact-matching overhead.

### 6.2 Query-Size Receptive Field Breakdown
We analyzed retrieval performance as a function of query fragment size ($|V_Q| \in [4, 12]$ atoms):

| Query Atoms ($|V_Q|$) | Order Embedding Recall | Order Embedding Precision | Mean True Hits in DB |
| :---: | :---: | :---: | :---: |
| **4** | **0.834** | 0.447 | 1,573.7 |
| **5** | **0.837** | 0.328 | 599.8 |
| **6** | 0.681 | 0.218 | 787.0 |
| **7** | 0.699 | 0.125 | 330.4 |
| **8** | 0.786 | 0.128 | 280.3 |
| **9** | 0.678 | 0.026 | 51.1 |
| **10** | **0.284** | 0.013 | 10.0 |
| **11** | 0.435 | 0.021 | 32.7 |
| **12** | **0.436** | 0.034 | 18.2 |

#### Physical & Architectural Reason for Degradation:
1. **Receptive Field Limitation:** With 4 `GINEConv` layers, a node's representation only aggregates information from its 4-hop neighborhood. For queries with 10–12 atoms containing fused ring systems (e.g., indole, biphenyl), the graph diameter exceeds the network's receptive field.
2. **Global Add Pooling Information Loss:** Sum-pooling compresses the multiset of node embeddings into a single fixed vector $z \in \mathbb{R}^{64}$. While 1-WL expressive, global sum pooling loses fine-grained subgraph relative positioning. As query size increases, the fraction of true database matches shrinks dramatically ($1{,}573$ hits for 4 atoms vs. $10$ hits for 10 atoms), exacerbating the needle-in-a-haystack challenge.

### 6.3 Functional Group Screening Performance
For standard chemist functional groups (aromatic rings, amides, esters, ketones), GNN B performed substantially better than on random subgraphs:
- Aromatic rings (benzene, pyridine, thiophene) achieved **$\ge 99.5\%$ recall**.
- Compact polar groups (amide, ketone, carboxylic acid) achieved **$\ge 96\%$ recall**.
Because functional groups are chemically recurrent and compact ($\le 6$ atoms), the learned embeddings encode their partial order accurately.

---

## 7. Division of Work & Team Contributions

- **Aayush Gupta (25m0140):**
  - Configured GPU acceleration environment and CUDA runtime optimization on RTX 3050 Laptop GPU.
  - Implemented PyTorch Geometric graph featurization pipeline (`atom_feat`, `bond_feat`, `mol_to_graph`).
  - Implemented Model A (`PairClassifier` with GINE backbone + BCE loss) and Model B (`OrderEmbedder` with non-negative projection + max-margin partial order loss).
  - Designed decision threshold tuning algorithms and high-recall cascade calibration routines.
  - Authored final project report and presentation script.

- **Srijal Gupta (25m0387):**
  - Downloaded and partitioned ZINC250k into disjoint training, validation, search database, and held-out query splits.
  - Built BFS-based connected subgraph query sampling (4 to 12 atoms) and defined 20 verified functional group queries.
  - Implemented hard-negative mining using RDKit exact checks against fingerprint screens.
  - Implemented RDKit classical baselines (`Chem.PatternFingerprint` bitwise screening + exact brute force timing harness).
  - Conducted query-size error ablation analysis and maintained project milestone tracking schedule.

- **Joint Work:** Cascaded search architecture design, experimental benchmarking, report editing, and video recording.

---

## 8. Conclusion & Future Directions

In this project, we designed and analyzed a Graph Neural Network substructure retrieval system evaluated against classical cheminformatics baselines. Our findings establish that:
1. **NeuroMatch Order Embeddings Outperform Siamese Classifiers:** Formulating subgraph containment geometrically as $z_Q \le z_T$ yields higher recall ($0.77$ vs $0.64$), faster search latency ($2.79\text{ ms}$ vs $6.66\text{ ms}$), and enables precomputed offline database embeddings.
2. **The Classical Baseline Remains Remarkably Strong:** RDKit's specialized 2048-bit `Chem.PatternFingerprint` screen combined with RDKit verification achieves $100\%$ precision, $100\%$ recall, and runs in **$5.61\text{ ms}$**—outperforming the GNN cascade ($20.21\text{ ms}$, $98.2\%$ recall).
3. **The Precision-Verification Bottleneck:** A learned filter cannot merely be fast; to beat classical cascades, it must achieve extreme precision at high recall to avoid drowning downstream verification in false positives.

### Future Improvements:
- **Hierarchical / Subgraph GNNs:** Incorporating subgraph-level message passing (e.g., Subgraph GNNs, cellular complexes) rather than whole-graph sum pooling to preserve cycle isomorphism for $>10$ atom queries.
- **Hybrid Neural-Fingerprint Filtering:** Combining bitwise fingerprint screening with GNN order embeddings in a triple-stage filter to eliminate hard negatives before exact matching.

---

## References

1. K. Xu, W. Hu, J. Leskovec, and S. Jegelka, *"How Powerful are Graph Neural Networks?"*, International Conference on Learning Representations (ICLR), 2019.
2. R. Ying, Z. Lou, J. You, C. Zheng, et al., *"Neural Subgraph Matching"*, Neural Information Processing Systems (NeurIPS), 2020.
3. D. Rose et al., *"PharmacoMatch: Efficient 3D Pharmacophore Screening via Neural Subgraph Matching"*, International Conference on Learning Representations (ICLR), 2025.
4. I. Vendrov, R. Kiros, S. Fidler, and R. Urtasun, *"Order-Embeddings of Images and Language"*, International Conference on Learning Representations (ICLR), 2016.
5. W. Hu, B. Liu, J. Gomes, P. Zitnik, et al., *"Strategies for Pre-training Graph Neural Networks"*, ICLR, 2020.
6. G. Landrum et al., *"RDKit: Open-source cheminformatics"*, https://github.com/rdkit/rdkit.
