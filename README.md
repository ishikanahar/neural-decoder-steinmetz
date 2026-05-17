# Neural Decoder for Behavioral Choice from Population Activity

Decoding left vs. right behavioral choices from neural population activity recorded during a visual discrimination task in mice.

**Dataset:** Steinmetz et al. (2019), DANDI Archive `000017`  
**Subject:** Mouse Richards — 778 neurons, 260 trials (Neuropixels recording, UCL)

---

## Scientific Question

Can we decode a mouse's behavioral choice (left vs. right wheel turn) from neural population activity recorded in a pre-response window (0–300ms post-stimulus)? And is the decoded structure meaningful — or just noise?

---

## Methods

### 1. Trial-Level Firing Rate Matrix
For each of 174 included trials (left or right choices only), firing rates were computed per neuron in a 300ms window following visual stimulus onset. This produced a **174 trials × 778 neurons** feature matrix.

### 2. PCA Dimensionality Reduction
Raw firing rate vectors are high-dimensional and highly correlated. PCA projects population activity onto orthogonal "neural modes" — axes of maximum variance that often capture behaviorally relevant structure.  
**Why not just use raw rates?** With 778 features and 174 trials, any decoder will overfit without dimensionality reduction.

### 3. Logistic Regression Decoder
Logistic regression with L2 regularization on the top 11 principal components. Evaluated via **5-fold stratified cross-validation** to ensure the decoder generalizes to held-out trials.

### 4. Permutation Testing
Labels shuffled 500 times; ROC-AUC recomputed each time. This builds a null distribution of decoder performance under no true signal — the standard validation used in neuroscience to confirm decoded structure is not artifactual.

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC (5-fold CV) | **0.884 ± 0.025** |
| Accuracy | 0.799 ± 0.039 |
| Permutation p-value | < 0.0001 |
| Null AUC mean | 0.505 ± 0.057 |

The real decoder (AUC = 0.884) vastly exceeds the permutation null distribution (mean = 0.505), confirming that the decoded structure reflects genuine neural representations of behavioral choice rather than chance classification.

Performance saturates around 11 principal components, suggesting most choice-related information is concentrated in a low-dimensional neural subspace.

---

## Repository Structure

```
neural-decoder-steinmetz/
├── decoder.py              # Full analysis pipeline
├── README.md               # This file
├── results_summary.json    # Key metrics
└── figures/
    └── decoder_results.png # All panels (A–F)
```

---

## Figure Panels

- **A** — PCA explained variance (individual + cumulative)
- **B** — Neural population geometry: PC1 vs PC2, colored by choice
- **C** — Decoder ROC-AUC as a function of number of PCs
- **D** — Permutation test: real decoder vs null distribution
- **E** — Per-fold cross-validation scores
- **F** — Confusion matrix (left vs right)

---

## How to Run

```bash
pip install pynwb scikit-learn matplotlib numpy pandas
python decoder.py
```

The script downloads nothing — it expects the NWB file at the path defined in `NWB_PATH`. Download via:

```python
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset('000017')
# pick smallest asset for fastest download
```

---

## Key Concepts 

**Why PCA before decoding?**  
Dimensionality reduction prevents overfitting, removes noise, and reveals the low-dimensional subspace where choice information lives. The fact that 11 PCs out of 778 neurons captures most signal tells us the population encodes choice in a structured, low-dimensional way — consistent with the "neural manifold" hypothesis.

**Why permutation testing?**  
Cross-validation estimates generalization but doesn't tell you if the result is above chance in a statistically rigorous way. Permutation tests build a true null distribution by destroying the label-feature relationship while preserving all other statistical structure.

**Why logistic regression and not a neural network?**  
With 174 trials, a neural network would massively overfit. Logistic regression with L2 regularization is interpretable, efficient, and well-matched to this regime. The decoder weights can also be projected back to neuron space to identify which neurons contribute most to choice decoding.

**What does ROC-AUC mean here?**  
AUC = 0.884 means that if you pick a random left-choice trial and a random right-choice trial, the decoder correctly ranks them 88.4% of the time. AUC = 0.5 is chance; AUC = 1.0 is perfect.
