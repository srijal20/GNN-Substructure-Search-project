# 2-Minute Video Presentation Script & Recording Guide
## CS 6012: Deep Learning — Open Project (Submission 2)
**Project Title:** GNN Molecular Substructure Search: Accuracy vs. Speed Trade-offs Against Classical Fingerprint Screening  
**Team Members:** Aayush Gupta (25m0140) & Srijal Gupta (25m0387)  
**Strict Video Constraint:** Length of video cannot exceed 2 minutes (120 seconds).  

---

### Video Structure & Speaking Split (Exact 120 Seconds)

```
0:00 - 0:30 (30s) -> Srijal: Problem Statement, Combinatorial Challenge & Baseline
0:30 - 1:05 (35s) -> Aayush: Architecture (GINEConv, NeuroMatch Order Embeddings & Cascade)
1:05 - 1:40 (35s) -> Srijal: Results Walkthrough, Verification Bottleneck & Error Breakdown
1:40 - 2:00 (20s) -> Aayush: Conclusion, Key Takeaway & Future Directions
```

---

### Minute-by-Minute Script

#### [0:00 – 0:30] Srijal Gupta — Problem & Motivation
*(Screen: Show README / Slide 1: Substructure search problem diagram showing query Q inside molecule T)*
> "Hello everyone. Our CS 6012 open project explores whether Graph Neural Networks can accelerate chemical substructure search compared to exact combinatorial algorithms like RDKit.  
> In chemical databases like ZINC with millions of compounds, chemists want to find molecules containing a query fragment Q. Exact subgraph isomorphism algorithms are NP-complete and scale poorly.  
> Industry uses 2048-bit Pattern Fingerprints, which guarantee 100% recall through bitwise containment. Our question is: can a learned deep GNN filter provide a better speed-accuracy trade-off?"

---

#### [0:30 – 1:05] Aayush Gupta — Model Architecture & Cascade
*(Screen: Show GNN_Substructure_Search.ipynb / Colab running on GPU, show GINEConv code and order loss formulation)*
> "To address this, we featurized molecules into bond-aware PyTorch Geometric graphs with 10 atom types and 4 bond types.  
> We implemented two models: first, a Siamese GIN with an MLP classifier, and second, a geometric order-embedder based on NeuroMatch.  
> In Model B, we map molecules into a non-negative cone such that Q is a subgraph of T if and only if z_Q is element-wise less than or equal to z_T.  
> All 50,000 database molecule embeddings are precomputed once. For queries, we use a two-stage cascade: the GNN shortlists candidates at 98% recall in under 3 milliseconds on our RTX 3050 GPU, and RDKit verifies only the shortlist, ensuring 100% final precision."

---

#### [1:05 – 1:40] Srijal Gupta — Results & The Verification Bottleneck
*(Screen: Show the Results Table and the Tradeoff Plot tradeoff.png from results/pilot/)*
> "Looking at the benchmark: brute-force RDKit takes 31.8 milliseconds per query.  
> Standalone, our GNN order embedding performs vector screening in just 2.79 milliseconds—an 11.4x retrieval speedup!  
> However, in the cascade, overall time is 20.2 milliseconds, while the classical Pattern Fingerprint baseline achieves 5.6 milliseconds.  
> The key scientific finding is the verification bottleneck: to keep 98% recall, the GNN returns 1,414 candidates versus 703 for fingerprints. Verifying those false positives in RDKit consumes most of the time.  
> Furthermore, query-size breakdown shows strong 83% recall for 4 to 5 atom fragments, but recall drops to 28% for 10-atom queries due to GIN receptive field limits."

---

#### [1:40 – 2:00] Aayush Gupta — Conclusion & Summary
*(Screen: Show Colab 1-click badge, GitHub repo with commits from both members, and final slide)*
> "In conclusion, while NeuroMatch order embeddings provide fast GPU vector search, classical Pattern Fingerprints remain remarkably strong because they produce tighter shortlists.  
> For deep learning to beat classical cheminformatics screening, future GNNs must improve precision on larger cycles.  
> All our code, Colab notebook, benchmark results, and formal report are available on our GitHub repository. Thank you!"

---

### Quick Recording Tips
- **Recording Tool:** You can use OBS Studio, Zoom (record with screen share), or Windows Game Bar (`Win + G`).
- **Screen setup:** Open the browser with 3 tabs:
  1. Slide deck or PDF report (`Final_Project_Report.pdf`)
  2. Colab notebook (`GNN_Substructure_Search.ipynb`) showing the GPU execution
  3. Tradeoff plot (`tradeoff.png`) in `results/pilot/`
- **Compression command (if video size is large):**
  ```bash
  ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4
  ```
