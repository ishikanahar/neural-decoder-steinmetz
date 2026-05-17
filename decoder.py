"""
Neural Decoder for Behavioral Choice from Population Activity
Steinmetz et al. (2019) Dataset — DANDI:000017

Scientific Question:
Can we decode a mouse's choice (left vs right vs no-go) from
the neural population activity recorded during a visual discrimination task?

Approach:
1. Load spike data from NWB file
2. Build trial-level firing rate matrices (neurons x trials)
3. Use PCA for dimensionality reduction
4. Train logistic regression decoder with cross-validation
5. Validate with permutation testing to distinguish real signal from chance
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import pynwb
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("Loading NWB data...")
NWB_PATH = "/home/claude/steinmetz_richards.nwb"
io = pynwb.NWBHDF5IO(NWB_PATH, 'r')
nwb = io.read()

trials_df = nwb.intervals['trials'].to_dataframe()
units_df   = nwb.units.to_dataframe()
n_units    = len(nwb.units)
print(f"  {len(trials_df)} trials | {n_units} neurons")

# ── 2. BUILD FIRING RATE MATRIX ───────────────────────────────────────────────
# Window: 0–300ms after visual stimulus onset (pre-response window)

WINDOW_START = 0.0   # seconds after stimulus
WINDOW_END   = 0.30  # seconds after stimulus

included_mask = trials_df['included'] & trials_df['response_choice'].isin([-1., 1.])
trials_clean  = trials_df[included_mask].copy()
print(f"  {len(trials_clean)} included trials (left/right choices only)")

labels = trials_clean['response_choice'].values  # -1 = left, 1 = right

firing_rates = np.zeros((len(trials_clean), n_units))

for unit_idx in range(n_units):
    spike_times = nwb.units['spike_times'][unit_idx]
    for trial_idx, (_, trial) in enumerate(trials_clean.iterrows()):
        stim_time  = trial['visual_stimulus_time']
        win_start  = stim_time + WINDOW_START
        win_end    = stim_time + WINDOW_END
        n_spikes   = np.sum((spike_times >= win_start) & (spike_times < win_end))
        firing_rates[trial_idx, unit_idx] = n_spikes / (WINDOW_END - WINDOW_START)

print(f"  Firing rate matrix: {firing_rates.shape} (trials x neurons)")
print(f"  Mean firing rate: {firing_rates.mean():.2f} Hz")

io.close()

# ── 3. PCA DIMENSIONALITY REDUCTION ──────────────────────────────────────────
# Why PCA? Neurons are highly correlated; PCA finds the axes of maximum
# variance in population activity — these "neural modes" often carry
# the most behaviorally relevant information.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(firing_rates)

pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
print(f"\nPCA: {n_components_90} PCs explain 90% of variance")

# Choose number of PCs to use for decoding
N_PCS = 11
X_decode = X_pca[:, :N_PCS]

# ── 4. DECODER WITH CROSS-VALIDATION ─────────────────────────────────────────
# Logistic regression with L2 regularization
# 5-fold stratified CV ensures balanced class distribution per fold

decoder = LogisticRegression(C=1.0, max_iter=500, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(decoder, X_decode, labels, cv=cv, scoring='roc_auc')

print(f"\nDecoder performance (5-fold CV):")
print(f"  ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Per-fold: {[f'{s:.3f}' for s in cv_scores]}")

# Also compute accuracy
acc_scores = cross_val_score(decoder, X_decode, labels, cv=cv, scoring='accuracy')
print(f"  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

# Fit final model for confusion matrix
decoder.fit(X_decode, labels)
y_pred = decoder.predict(X_decode)

# ── 5. PERMUTATION TESTING ───────────────────────────────────────────────────
# Key validation: shuffle labels 500 times and recompute ROC-AUC.
# If the real decoder beats the null distribution, our result is meaningful.
# This is the rigorous way to distinguish real neural structure from
# overfitting or chance classification.

print("\nRunning permutation test (500 permutations)...")
N_PERMS = 500
perm_scores = np.zeros(N_PERMS)
rng = np.random.default_rng(42)

for i in range(N_PERMS):
    labels_perm = rng.permutation(labels)
    perm_fold_scores = cross_val_score(decoder, X_decode, labels_perm, cv=cv, scoring='roc_auc')
    perm_scores[i] = perm_fold_scores.mean()

real_score = cv_scores.mean()
p_value = np.mean(perm_scores >= real_score)
print(f"  Real ROC-AUC:  {real_score:.3f}")
print(f"  Null mean:     {perm_scores.mean():.3f} ± {perm_scores.std():.3f}")
print(f"  p-value:       {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")

# ── 6. DECODER ACROSS PC DIMENSIONS ──────────────────────────────────────────
# How does performance change as we add more PCs?
# This reveals how much signal each additional dimension adds.

n_pc_range = list(range(1, 21))
auc_by_pc  = []
for n in n_pc_range:
    X_n = X_pca[:, :n]
    s = cross_val_score(decoder, X_n, labels, cv=cv, scoring='roc_auc')
    auc_by_pc.append(s.mean())

print(f"\nDecoder performance by PC count:")
for n, s in zip(n_pc_range, auc_by_pc):
    print(f"  {n:2d} PCs: ROC-AUC = {s:.3f}")

# ── 7. FIGURES ────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Neural Decoder for Behavioral Choice\nSteinmetz et al. (2019) — Mouse Visual Discrimination",
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Fig A: PCA explained variance ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(range(1, 21), explained_var[:20] * 100, color='steelblue', alpha=0.7, label='Individual')
ax1.plot(range(1, 21), cumulative_var[:20] * 100, 'r-o', ms=4, lw=1.5, label='Cumulative')
ax1.axhline(90, color='gray', linestyle='--', lw=1, alpha=0.7)
ax1.text(20.5, 90.5, '90%', va='bottom', ha='left', fontsize=8, color='gray')
ax1.set_xlabel('Principal Component', fontsize=10)
ax1.set_ylabel('Explained Variance (%)', fontsize=10)
ax1.set_title('A   PCA: Explained Variance', fontsize=10, fontweight='bold', loc='left')
ax1.legend(fontsize=8)
ax1.set_xlim(0.5, 20.5)

# ── Fig B: PCA scatter — PC1 vs PC2 colored by choice ──
ax2 = fig.add_subplot(gs[0, 1])
colors = {-1.: '#E74C3C', 1.: '#3498DB'}
choice_labels_str = {-1.: 'Left choice', 1.: 'Right choice'}
for choice in [-1., 1.]:
    mask = labels == choice
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[choice], alpha=0.55, s=30, label=choice_labels_str[choice])
ax2.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=10)
ax2.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=10)
ax2.set_title('B   Neural Population Geometry\n(PC1 vs PC2)', fontsize=10, fontweight='bold', loc='left')
ax2.legend(fontsize=8)

# ── Fig C: Decoder performance by PC count ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(n_pc_range, auc_by_pc, 'o-', color='steelblue', lw=2, ms=5)
ax3.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.7, label='Chance (0.50)')
ax3.fill_between(n_pc_range, 0.5, auc_by_pc, alpha=0.15, color='steelblue')
ax3.set_xlabel('Number of PCs', fontsize=10)
ax3.set_ylabel('ROC-AUC (5-fold CV)', fontsize=10)
ax3.set_title('C   Decoder Performance vs\nDimensionality', fontsize=10, fontweight='bold', loc='left')
ax3.set_ylim(0.45, 1.0)
ax3.legend(fontsize=8)

# ── Fig D: Permutation test ──
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(perm_scores, bins=30, color='gray', alpha=0.7, label='Permutation null')
ax4.axvline(real_score, color='#E74C3C', lw=2.5, label=f'Real decoder\nAUC={real_score:.3f}')
ax4.axvline(perm_scores.mean(), color='black', lw=1, linestyle='--', alpha=0.6, label='Null mean')
ax4.set_xlabel('ROC-AUC', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title(f'D   Permutation Test\np = {p_value:.4f}', fontsize=10, fontweight='bold', loc='left')
ax4.legend(fontsize=7.5)

# ── Fig E: Cross-validation fold scores ──
ax5 = fig.add_subplot(gs[1, 1])
fold_nums = np.arange(1, 6)
bar_colors = ['#2ECC71' if s > 0.5 else '#E74C3C' for s in cv_scores]
bars = ax5.bar(fold_nums, cv_scores, color=bar_colors, alpha=0.8, edgecolor='white')
ax5.axhline(cv_scores.mean(), color='steelblue', lw=2, linestyle='--', label=f'Mean={cv_scores.mean():.3f}')
ax5.axhline(0.5, color='gray', lw=1, linestyle=':', alpha=0.7, label='Chance')
ax5.set_xlabel('CV Fold', fontsize=10)
ax5.set_ylabel('ROC-AUC', fontsize=10)
ax5.set_title('E   Cross-Validation Fold Scores', fontsize=10, fontweight='bold', loc='left')
ax5.set_ylim(0.4, 1.0)
ax5.set_xticks(fold_nums)
ax5.legend(fontsize=8)

# ── Fig F: Confusion matrix ──
ax6 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(labels, y_pred, labels=[-1., 1.])
disp = ConfusionMatrixDisplay(cm, display_labels=['Left', 'Right'])
disp.plot(ax=ax6, colorbar=False, cmap='Blues')
ax6.set_title('F   Confusion Matrix\n(Full Dataset Fit)', fontsize=10, fontweight='bold', loc='left')

plt.savefig('/home/claude/neural-decoder-steinmetz/figures/decoder_results.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nFigures saved.")

# ── 8. SAVE SUMMARY ───────────────────────────────────────────────────────────

summary = {
    'n_trials': int(len(trials_clean)),
    'n_neurons': int(n_units),
    'n_pcs_used': int(N_PCS),
    'pcs_for_90pct': int(n_components_90),
    'roc_auc_mean': float(cv_scores.mean()),
    'roc_auc_std': float(cv_scores.std()),
    'accuracy_mean': float(acc_scores.mean()),
    'accuracy_std': float(acc_scores.std()),
    'permutation_p_value': float(p_value),
    'null_auc_mean': float(perm_scores.mean()),
    'null_auc_std': float(perm_scores.std()),
}

import json
with open('/home/claude/neural-decoder-steinmetz/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n=== FINAL SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

# ── 9. TEMPORAL DECODER ───────────────────────────────────────────────────────
# Instead of one 300ms window, decode from 50ms sliding bins across time.
# This shows HOW the choice signal evolves — when does the population
# "know" the choice? This is how real BCI decoders track intention over time.

print("\n=== TEMPORAL DECODER ===")

BIN_SIZE   = 0.050   # 50ms bins
T_START    = -0.20   # 200ms before stimulus
T_END      =  0.50   # 500ms after stimulus
bin_edges  = np.arange(T_START, T_END, BIN_SIZE)
bin_centers = bin_edges[:-1] + BIN_SIZE / 2
n_bins = len(bin_centers)

io2 = pynwb.NWBHDF5IO(NWB_PATH, 'r')
nwb2 = io2.read()

temporal_fr = np.zeros((len(trials_clean), n_units, n_bins))

for unit_idx in range(n_units):
    spike_times = nwb2.units['spike_times'][unit_idx]
    for trial_idx, (_, trial) in enumerate(trials_clean.iterrows()):
        stim_time = trial['visual_stimulus_time']
        for b, t in enumerate(bin_edges[:-1]):
            win_s = stim_time + t
            win_e = stim_time + t + BIN_SIZE
            n_sp  = np.sum((spike_times >= win_s) & (spike_times < win_e))
            temporal_fr[trial_idx, unit_idx, b] = n_sp / BIN_SIZE

io2.close()
print(f"  Temporal matrix: {temporal_fr.shape} (trials x neurons x bins)")

# Decode at each time bin
temporal_auc = np.zeros(n_bins)
temporal_err = np.zeros(n_bins)

for b in range(n_bins):
    X_bin = temporal_fr[:, :, b]
    X_bin_scaled = StandardScaler().fit_transform(X_bin)
    pca_t = PCA(n_components=11, random_state=42)
    X_bin_pca = pca_t.fit_transform(X_bin_scaled)
    dec_t = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    scores = cross_val_score(dec_t, X_bin_pca, labels, cv=cv, scoring='roc_auc')
    temporal_auc[b] = scores.mean()
    temporal_err[b] = scores.std()

print(f"  Peak AUC: {temporal_auc.max():.3f} at t={bin_centers[temporal_auc.argmax()]*1000:.0f}ms")
print(f"  AUC at stimulus onset (t=0): {temporal_auc[np.argmin(np.abs(bin_centers))]:.3f}")

# ── 10. FEATURE IMPORTANCE ────────────────────────────────────────────────────
# Project logistic regression weights back into neuron space.
# decoder_weights (in PC space) × PCA components = neuron-level contributions.
# This tells us WHICH neurons drive the choice decoding signal.

print("\n=== FEATURE IMPORTANCE ===")

# Refit on full data with 11 PCs
scaler_fi = StandardScaler()
X_fi = scaler_fi.fit_transform(firing_rates)
pca_fi = PCA(n_components=11, random_state=42)
X_fi_pca = pca_fi.fit_transform(X_fi)

dec_fi = LogisticRegression(C=1.0, max_iter=500, random_state=42)
dec_fi.fit(X_fi_pca, labels)

# Project back: shape (n_neurons,)
neuron_weights = dec_fi.coef_[0] @ pca_fi.components_
abs_weights    = np.abs(neuron_weights)

top_idx    = np.argsort(abs_weights)[::-1][:15]
bottom_idx = np.argsort(abs_weights)[:15]

print(f"  Top 15 neurons (highest |weight|): indices {top_idx[:5]}...")
print(f"  Weight range: {neuron_weights.min():.3f} to {neuron_weights.max():.3f}")

# Firing rate traces for top 5 neurons across time, split by choice
top5 = top_idx[:5]
left_trials  = labels == -1.
right_trials = labels ==  1.

# ── 11. COMBINED FIGURES ──────────────────────────────────────────────────────

fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle("BCI Decoder Extensions: Temporal Dynamics & Neural Feature Importance\nSteinmetz et al. (2019)",
              fontsize=13, fontweight='bold', y=0.99)

# ── Panel A: Temporal decoding curve ──
ax = axes[0, 0]
ax.plot(bin_centers * 1000, temporal_auc, 'steelblue', lw=2.5)
ax.fill_between(bin_centers * 1000,
                temporal_auc - temporal_err,
                temporal_auc + temporal_err,
                alpha=0.2, color='steelblue')
ax.axhline(0.5, color='gray', lw=1, linestyle='--', label='Chance')
ax.axvline(0, color='black', lw=1.5, linestyle=':', label='Stimulus onset')
peak_t = bin_centers[temporal_auc.argmax()] * 1000
ax.axvline(peak_t, color='#E74C3C', lw=1.5, linestyle='--',
           label=f'Peak ({peak_t:.0f}ms)')
ax.set_xlabel('Time from Stimulus Onset (ms)', fontsize=10)
ax.set_ylabel('ROC-AUC (5-fold CV)', fontsize=10)
ax.set_title('A   Temporal Decoding Curve\n(50ms sliding bins)', fontsize=10, fontweight='bold', loc='left')
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=8)

# ── Panel B: Neuron weight distribution ──
ax = axes[0, 1]
ax.hist(neuron_weights, bins=50, color='steelblue', alpha=0.75, edgecolor='white')
ax.axvline(0, color='black', lw=1)
for w in neuron_weights[top_idx[:5]]:
    ax.axvline(w, color='#E74C3C', lw=1.2, alpha=0.7)
ax.set_xlabel('Decoder Weight (projected to neuron space)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('B   Neuron Weight Distribution\n(red = top 5 neurons)', fontsize=10, fontweight='bold', loc='left')

# ── Panel C: Top 15 neuron weights ──
ax = axes[0, 2]
top15_w = neuron_weights[top_idx]
colors_bar = ['#E74C3C' if w > 0 else '#3498DB' for w in top15_w]
ax.barh(range(15), top15_w[::-1], color=colors_bar[::-1], alpha=0.8)
ax.set_yticks(range(15))
ax.set_yticklabels([f'Neuron {top_idx[14-i]}' for i in range(15)], fontsize=7.5)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Decoder Weight', fontsize=10)
ax.set_title('C   Top 15 Neurons by\n|Decoder Weight|', fontsize=10, fontweight='bold', loc='left')

# ── Panel D–F: Firing rate traces for top 3 neurons ──
for panel_i, unit_i in enumerate(top5[:3]):
    ax = axes[1, panel_i]
    # Mean firing rate across time bins for left vs right trials
    fr_left  = temporal_fr[left_trials,  unit_i, :].mean(axis=0)
    fr_right = temporal_fr[right_trials, unit_i, :].mean(axis=0)
    sem_left  = temporal_fr[left_trials,  unit_i, :].std(axis=0) / np.sqrt(left_trials.sum())
    sem_right = temporal_fr[right_trials, unit_i, :].std(axis=0) / np.sqrt(right_trials.sum())

    t_ms = bin_centers * 1000
    ax.plot(t_ms, fr_left,  '#E74C3C', lw=2, label='Left choice')
    ax.plot(t_ms, fr_right, '#3498DB', lw=2, label='Right choice')
    ax.fill_between(t_ms, fr_left  - sem_left,  fr_left  + sem_left,  alpha=0.15, color='#E74C3C')
    ax.fill_between(t_ms, fr_right - sem_right, fr_right + sem_right, alpha=0.15, color='#3498DB')
    ax.axvline(0, color='black', lw=1.2, linestyle=':', alpha=0.7)
    ax.set_xlabel('Time from Stimulus (ms)', fontsize=9)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=9)
    panel_label = chr(ord('D') + panel_i)
    ax.set_title(f'{panel_label}   Neuron {unit_i} (rank #{panel_i+1})\n±SEM across trials',
                 fontsize=10, fontweight='bold', loc='left')
    if panel_i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/neural-decoder-steinmetz/figures/temporal_importance.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nExtended figures saved.")
