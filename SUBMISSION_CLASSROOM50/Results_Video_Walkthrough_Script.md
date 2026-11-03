# 2-Minute Results Video Script (Simplified & Easy to Speak)
## CS 6012: Deep Learning — Open Project (Submission 2)
**Project:** GNN Molecular Substructure Search  
**Speakers:** Srijal Gupta (25m0140/25m0387) & Aayush Gupta  
**Total Target Time:** ~1 minute 45 seconds to 2 minutes (approx. 240 words)  

---

### Part 1 [0:00 – 0:30] Srijal Gupta: Problem & Idea
*(Screen: Show README.md or slide with the query fragment inside a molecule)*

> "Hello everyone. Our project explores using Graph Neural Networks for molecular substructure search.  
> 
> In drug discovery, chemists search millions of molecules to find if a query fragment $Q$ is present inside a target molecule $T$.  
> 
> Exact graph matching tools like RDKit are accurate, but they become slow on large databases.  
> 
> The standard industry baseline uses 2048-bit Pattern Fingerprints, which filter molecules using fast bitwise checks.  
> 
> Our goal was to see if a learned GNN could act as a faster filter without missing valid molecules."

---

### Part 2 [0:30 – 1:00] Aayush Gupta: Models & Two-Stage Cascade
*(Screen: Show Colab notebook or architecture diagram)*

> "We tested two GNN architectures on the ZINC250k chemical library:  
> First, a Siamese GIN model with an MLP classifier.  
> And second, a NeuroMatch order-embedding model.  
> 
> In the order-embedding model, if $Q$ is a subgraph of $T$, its embedding vector $z_Q$ is element-wise smaller than $z_T$.  
> 
> Because of this, we precompute all database molecule embeddings once. Searching becomes a single matrix check on GPU in under 3 milliseconds!  
> 
> To guarantee zero false positives, we use a two-stage cascade:  
> The GNN shortlists likely matches at 98% recall, and RDKit verifies only that shortlist."

---

### Part 3 [1:00 – 1:35] Srijal Gupta: Results & The Key Bottleneck
*(Screen: Show the Results Table and the Tradeoff Plot `tradeoff.png`)*

> "Looking at our results on 5,000 molecules:  
> - Brute-force RDKit takes **31.8 milliseconds** per query.  
> - Our GNN order embedding alone takes just **2.8 milliseconds**—an **11x speedup**!  
> - But when combined with RDKit verification, the cascade takes **20.2 milliseconds**.  
> - Meanwhile, the classical Pattern Fingerprint baseline runs in just **5.6 milliseconds** with 100% accuracy.  
> 
> Why does this happen? We call it the **Verification Bottleneck**:  
> To keep 98% recall, the GNN returns 1,414 candidates—twice as many as fingerprints.  
> Verifying those extra false positives in RDKit consumes 17 milliseconds of CPU time!  
> 
> Also, query size matters: recall is 83% on small 4-atom fragments, but drops to 28% on 10-atom rings because 4 GIN layers cannot see the entire cycle."

---

### Part 4 [1:35 – 2:00] Aayush Gupta: Summary & Conclusion
*(Screen: Show GitHub Repo with commit history & Colab link)*

> "To conclude:  
> 1. GNN order embeddings are fast and effective for candidate retrieval.  
> 2. However, classical Pattern Fingerprints remain the winning choice in practice because they produce far fewer false positives.  
> 3. For neural methods to beat classical baselines, they must achieve much higher precision at high recall.  
> 
> All our code, Colab notebook, and final report are available on our GitHub repository. Thank you!"

---

### Speaking Notes:
- Keep a relaxed, conversational tone.
- No complicated jargon.
- Pauses at the bullet points make it very easy to follow for the graders.
- Total word count is ~240 words, which naturally takes **1 minute 45 seconds to 2 minutes**.
