# CS 6012: Results Video Walkthrough Script (2 to 3 Minutes)
## Project: GNN Molecular Substructure Search (Submission 2)
**Presenters:** Aayush Gupta (25m0140) & Srijal Gupta (25m0387)  
**Total Target Duration:** 2 minutes 30 seconds (150 seconds)  

---

### Visual Setup (What to display on screen)
Prepare 3 tabs or slides in advance:
- **Screen 1 (0:00 – 0:30):** Title slide or `README.md` showing the Two-Stage Cascade Architecture diagram.
- **Screen 2 (0:30 – 1:15):** Benchmark Comparison Table & `training_curves.png` (from `results/pilot/`).
- **Screen 3 (1:15 – 2:00):** Tradeoff Plot (`tradeoff.png`) & Query Size Breakdown (`order_model_by_query_size.csv`).
- **Screen 4 (2:00 – 2:30):** Colab notebook (`GNN_Substructure_Search.ipynb`) with 1-click badge & GitHub repo.

---

### Minute-by-Minute Speaking Script

```
0:00 - 0:35 (35s) -> Srijal: Problem Formulation & Evaluated Methods
0:35 - 1:15 (40s) -> Aayush: Core Benchmark Numbers & Speedup Analysis
1:15 - 1:55 (40s) -> Srijal: The Verification Bottleneck & Query Size Breakdown
1:55 - 2:30 (35s) -> Aayush: Key Scientific Takeaways & Conclusion
```

---

#### Part 1: Problem Formulation & Evaluated Methods [0:00 – 0:35]
**Speaker: Srijal Gupta**  
*(Visual: Show Title & Two-Stage Cascade Diagram on screen)*

> "Hello everyone. In this video, we present the empirical results of our CS 6012 open project on GNN-based Molecular Substructure Search.  
> 
> Exact chemical substructure search using combinatorial algorithms like RDKit’s Ullmann or VF2 is computationally expensive across large databases because it scales exponentially in the worst case.  
> 
> To test whether deep learning can accelerate this, we evaluated five distinct methods on the ZINC250k chemical library:  
> 1. Brute-force RDKit exact matching as ground truth,  
> 2. The industrial baseline: RDKit’s 2048-bit Pattern Fingerprint bitwise screen,  
> 3. Model A: a Siamese bond-aware GIN with an MLP classifier,  
> 4. Model B: NeuroMatch-style geometric order embeddings ($z_Q \le z_T$ in $\mathbb{R}_{\ge 0}^{64}$), and  
> 5. A Two-Stage Cascade, where the GNN shortlists candidate molecules at 98% recall, and RDKit verifies only that shortlist."

---

#### Part 2: Core Benchmark Numbers & Speedup Analysis [0:35 – 1:15]
**Speaker: Aayush Gupta**  
*(Visual: Switch screen to the Benchmark Results Table & Training Curves)*

> "Here are our primary quantitative results evaluated over 50,000 database molecules and 120+ test queries:  
> 
> - Brute-force RDKit takes **31.8 milliseconds** per query.  
> - Our Model B order embedding alone performs candidate screening in just **2.79 milliseconds**, achieving an **11.4x speedup** on our RTX 3050 GPU!  
> - However, looking at retrieval quality, GNN B alone achieves **77.1% recall** and **27% precision**, whereas the classical Pattern Fingerprint screen achieves **100% recall** and **70.3% precision** in just **1.09 milliseconds**.  
> - When we place both into a practical cascade where RDKit verifies the candidates:  
>   - The classical Pattern Fingerprint cascade runs in **5.61 milliseconds** with 100% precision and 100% recall ($5.7\times$ speedup).  
>   - The GNN B cascade runs in **20.21 milliseconds** with 98.2% recall ($1.6\times$ speedup)."

---

#### Part 3: The Verification Bottleneck & Query Size Breakdown [1:15 – 1:55]
**Speaker: Srijal Gupta**  
*(Visual: Switch screen to `tradeoff.png` Pareto curve and the query-size table)*

> "This reveals the central technical finding of our project: **The Verification Bottleneck**.  
> 
> Even though GNN vector screening takes under 3 milliseconds, maintaining 98% recall forces the threshold to be relaxed, causing GNN B to return an average of **1,414 candidate molecules** per query—nearly double the 703 candidates returned by Pattern Fingerprints.  
> 
> Since verifying a false candidate in RDKit takes about 0.012 milliseconds, checking those extra false positives consumes **17 milliseconds of CPU time**, which dominates the entire cascade latency!  
> 
> Furthermore, looking at our query-size ablation:  
> - On small 4 to 5 atom subgraphs, the GNN achieves high recall of **83.4% to 83.7%**.  
> - But on larger 10 to 12 atom subgraphs, recall drops sharply to **28% to 44%**.  
> This happens because our 4-layer GINE receptive field and whole-graph sum-pooling struggle to preserve multi-hop cycle topologies as molecular diameter expands."

---

#### Part 4: Key Takeaways & Conclusion [1:55 – 2:30]
**Speaker: Aayush Gupta**  
*(Visual: Switch screen to Colab Notebook / GitHub Repository showing code, commits, and tag `v1.0`)*

> "In conclusion, our research delivers three key insights:  
> 
> 1. **Order embeddings work:** NeuroMatch partial-order embeddings significantly outperform Siamese classification heads, reaching higher recall (0.77 vs 0.64) and enabling fast one-time offline database pre-embedding.  
> 2. **Classical baselines are formidable:** RDKit's Pattern Fingerprints remain the superior practical filter because their tight bitwise containment produces far fewer false positives.  
> 3. **The Lesson for Neural Retrieval:** To beat classical cheminformatics screening, future GNNs cannot just be fast—they must achieve near-perfect precision at the 98% recall boundary to avoid drowning downstream verification.  
> 
> All our code, interactive Colab notebook, benchmark CSVs, and formal report are fully documented on GitHub. Thank you!"

---

### Tips for Recording
1. **Audio Clarity:** Use headphones with a mic. Speak calmly at a steady, conversational pace.
2. **Smooth Screen Switching:** Use OBS Studio with window capture, or record a Zoom meeting with screenshare.
3. **Pacing Check:** 
   - 0:35 mark $\to$ switch from Title to Table
   - 1:15 mark $\to$ switch from Table to Tradeoff Plot
   - 1:55 mark $\to$ switch to Colab notebook / GitHub
4. **Target Time:** Read at normal speed, this script takes between **2 minutes 15 seconds and 2 minutes 45 seconds**, perfectly within the target 2–3 minute window!
