================================================================================
CS 6012: Deep Learning — Autumn 2026 (IIT Bombay)
Open Project: Submission 2 (Code-Based Project)
Final Submission Package for Classroom50 & Moodle
================================================================================

PROJECT TITLE:
GNN Molecular Substructure Search: Accuracy vs. Speed Trade-offs Against Classical Fingerprint Screening

AUTHORS / TEAM MEMBERS:
1. Aayush Gupta  | Roll Number: 25m0140 | Email: aayushg416@gmail.com / 25m0140@iitb.ac.in
2. Srijal Gupta  | Roll Number: 25m0387 | Email: sirjalgupta@gmail.com / 25m0387@iitb.ac.in

PRIMARY GITHUB REPOSITORY:
https://github.com/srijal20/GNN-Substructure-Search-project

================================================================================
1. HONOR CODE DECLARATION
================================================================================
We, Aayush Gupta (25m0140) and Srijal Gupta (25m0387), solemnly declare that this 
submitted project work is our own, except where explicitly cited in our formal 
project report. All external open-source chemical and graph libraries (PyTorch 
Geometric, RDKit, ReportLab) and LLM-assisted interactions have been openly 
acknowledged, documented, and cited according to the academic integrity policies 
of the Indian Institute of Technology Bombay.

Signed:
- Aayush Gupta (25m0140)
- Srijal Gupta (25m0387)

================================================================================
WHAT IS THIS PROJECT ABOUT? (EXPLAINED IN 60 SECONDS LIKE A 10TH GRADER)
================================================================================
The Lego Toy Analogy:
Imagine you have 50,000 big Lego models (drug molecules like Aspirin, Paracetamol).
Someone hands you a small Lego piece (a query fragment, e.g. a Pyridine ring) and 
asks: "Which of our 50,000 toys contain this exact piece embedded inside them?"

In chemistry, this is called Substructure Search. It is NP-complete (checking atom
by atom takes exponential time). We compared 4 ways to solve this:

1. RDKit Exact Search (Brute Force):
   Checks atom-by-atom under a microscope. 100% correct, but slow on big databases.
2. Pattern Fingerprints (The 2048-Bit Barcode):
   Fast checklist. If the query piece has a feature missing in the big molecule, 
   drop it immediately in 1.0 ms! (Classical industrial standard).
3. Graph Neural Network (AI Order Embedding):
   Turns molecules into numbers (z in R+^64). If query is inside molecule, then 
   z_query <= z_molecule. Instant GPU vector check in 2.8 ms!
4. Two-Stage Cascade (The Airport Security Combo):
   Guard 1 (GNN) does a fast metal detector scan to shortlist candidates, and 
   Guard 2 (RDKit) double-checks only the suspects to guarantee 100% accuracy.

For the full simple walkthrough, see: 'PROJECT_EXPLAINED_SUPER_SIMPLE.md'

================================================================================
2. LINKS TO CHATGPT / EQUIVALENT LLMs & PROMPT ARCHIVE
================================================================================
In accordance with course guidelines, we openly disclose all AI assistant / LLM 
interactions used during project ideation, debugging, and visualization polish:

1. Claude (Anthropic) Ideation & Proposal Form Discussion:
   https://claude.ai/share/dca653e6-1d40-4953-bd89-eec3ecff4f49
   Summary: Used for initial problem brainstorming, vetting chemical validity of 
   substructure screening, correcting misconceptions about Morgan fingerprints vs. 
   RDKit Pattern Fingerprints, and structuring the team proposal.

2. Antigravity Agent (Google DeepMind) Pair-Programming Transcript:
   Summary: Used for CUDA runtime debugging on NVIDIA GeForce RTX 3050, PyG batching 
   optimization, ReportLab academic PDF typesetting, resolving Matplotlib Pareto plot 
   text overlaps via callouts, and authoring the 2-minute video presentation script.
   Key prompt themes included:
   - "How to enforce guaranteed 100% recall bitwise containment baseline using RDKit PatternFingerprint?"
   - "Implementation of NeuroMatch partial order violation loss E(q, t) = ||max(0, zq - zt)||^2 in PyTorch."
   - "Vectorized NumPy SIMD screening over 50,000 molecules with packed uint8 arrays."

================================================================================
3. CANNED DEMO FOR TEACHING TEAM / TAs (MINIMAL EFFORT EXECUTION)
================================================================================
We have designed a self-contained, 30-second canned demonstration script so the 
teaching team can evaluate our retrieval pipeline with zero friction:

Option A (1-Click Google Colab — No Local Setup):
   Open the interactive Colab notebook via the 1-click badge:
   https://colab.research.google.com/github/srijal20/GNN-Substructure-Search-project/blob/main/GNN_Substructure_Search.ipynb
   Select Runtime > Run all (runs on free T4 GPU).

Option B (Local Canned Demo — Takes ~10 to 15 seconds):
   Windows: Double-click 'run_demo.bat' or run:
     python canned_demo.py
   Linux / Mac:
     bash run_demo.sh  or  python3 canned_demo.py

   What this demo does:
   - Loads a curated sample of 50 diverse drug molecules (Aspirin, Caffeine, Imatinib, etc.).
   - Runs a query fragment (e.g. Pyridine ring: 'c1ccncc1').
   - Displays side-by-side comparative table:
     * RDKit Exact Ground Truth (combinatorial matching)
     * Classical Pattern Fingerprint Screen
     * GNN Order Embedding candidate screening
     * Two-Stage Cascade (GNN shortlist + RDKit verification)
   - Outputs candidates shortlisted, verified matches, and latencies in milliseconds.

Option C (Full Scientific Benchmark Reproduction):
   python gnn_substructure_search.py
   (Or run the complete 'GNN_Substructure_Search.ipynb' notebook)

================================================================================
4. (VIDEO) PRESENTATION (2 TO 3 MINUTES)
================================================================================
- Target Duration: strictly bounded between 1m45s and 2m30s.
- Detailed Walkthrough Script: see 'Results_Video_Walkthrough_Script.md'
- Speakers: Srijal Gupta (Problem Formulation, Trade-off Plot & Verification Bottleneck) 
  and Aayush Gupta (Two-Stage Architecture, Quantitative Benchmarks & Scientific Takeaways).
- Submission Link & File: See 'VIDEO_SUBMISSION_LINK.txt' and Moodle video upload.

================================================================================
5. PROJECT REPORT (PDF DOCUMENT)
================================================================================
- File: 'Final_Project_Report.pdf' (Strict 2-page academic paper).
- Markdown Source: 'Final_Project_Report.md'.
- Contents: Abstract, Problem Formulation, Classical Fingerprints vs. GNN Architectures, 
  Two-Stage Cascade, Pareto Frontier, The Verification Bottleneck Analysis, Query-Size 
  Receptive Field Breakdown, and Conclusion.

================================================================================
6. CODE COMMENTARY & CODEBASE CONTENTS
================================================================================
Both 'GNN_Substructure_Search.ipynb' and 'gnn_substructure_search.py' contain 
rich, line-by-line pedagogical comments designed so that anyone reading the code 
for the first time will understand:
  - Molecular graph tensors (atoms = nodes, bonds = edges, element one-hot vectors).
  - Why GIN sum-aggregation matches the 1-Weisfeiler-Lehman (1-WL) isomorphism test.
  - Partial order embeddings in the non-negative orthant R^d >= 0.
  - Hard negative mining (passing fingerprint filters, failing exact graph match).
  - Offline pre-embedding vs. online vector screening latency.

================================================================================
PACKAGE FILE INDEX:
================================================================================
- README.txt                         : This file (Honor Code, LLM Links, TA instructions).
- README.md                          : Polished GitHub documentation with Colab badge.
- canned_demo.py                     : 30-second evaluation demo script for TAs.
- run_demo.bat                       : 1-click Windows launcher for canned demo.
- run_demo.sh                        : 1-click Linux/Mac launcher for canned demo.
- Final_Project_Report.pdf           : Formal 2-page project report PDF.
- Final_Project_Report.md            : Full report source in Markdown.
- GNN_Substructure_Search.ipynb      : Interactive Colab notebook with rich comments.
- gnn_substructure_search.py         : Complete Python pipeline.
- Results_Video_Walkthrough_Script.md: Simplified 2-minute video presentation script.
- VIDEO_SUBMISSION_LINK.txt          : Video details and submission link reference.
- requirements.txt                   : Environment dependencies.
- Updated_Schedule_Spreadsheet.pdf   : 1-page milestone tracking spreadsheet (all completed).
- Checkpoint4_Status_Report.pdf      : 1-page Checkpoint #4 status report PDF.
- results/                           : Trade-off plot (tradeoff.png), curves, and CSV tables.
================================================================================
