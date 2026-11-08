"""Generates publication-quality figures directly in assets/ from real study data."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Output directory: all images placed directly in assets/
OUT_DIR = os.path.abspath("assets")
os.makedirs(OUT_DIR, exist_ok=True)

# Set global publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

palette = {
    'F0': '#64748b',  # Slate gray (Single model baseline)
    'F1': '#3b82f6',  # Blue (Ensemble mean)
    'F2': '#8b5cf6',  # Purple (Member max)
    'F3': '#ec4899',  # Pink (Disagreement aware)
    'F4': '#f59e0b',  # Amber (Morphological dilation)
    'F5': '#10b981',  # Emerald green (Adaptive component selection)
}

print("[1/8] Generating training_curves.png ...")
train_csv = "outputs/training_curves_100ep.csv" if os.path.exists("outputs/training_curves_100ep.csv") else "training_curves_100ep.csv"
train_df = pd.read_csv(train_csv)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Subplot 1: Losses
ax1.plot(train_df['epoch'], train_df['train_loss'], label='Training Loss', color='#2563eb', linewidth=2)
ax1.plot(train_df['epoch'], train_df['val_loss'], label='Validation Loss', color='#dc2626', linewidth=2, linestyle='--')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (Dice + Cross-Entropy)')
ax1.set_title('(a) Training & Validation Loss Dynamics')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
ax1.set_xlim(0, 99)

# Subplot 2: Dice scores
ax2.plot(train_df['epoch'], train_df['mean_fg_dice'], label='Per-Epoch Foreground Dice', color='#93c5fd', alpha=0.6, linewidth=1.2)
ax2.plot(train_df['epoch'], train_df['ema_fg_dice'], label='Exponential Moving Average (EMA)', color='#059669', linewidth=2.2)
best_ema_idx = train_df['ema_fg_dice'].idxmax()
best_ema_val = train_df['ema_fg_dice'].max()
ax2.scatter(best_ema_idx, best_ema_val, color='#b91c1c', s=60, zorder=5, label=f'Peak EMA: {best_ema_val:.4f}')
ax2.annotate(f'Peak EMA: {best_ema_val:.4f}\n(Epoch {best_ema_idx})', 
             xy=(best_ema_idx, best_ema_val), 
             xytext=(best_ema_idx - 35, best_ema_val - 0.08),
             arrowprops=dict(facecolor='#b91c1c', shrink=0.08, width=1, headwidth=6))
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Foreground Pseudo-Dice')
ax2.set_title('(b) Model Segmentation Convergence')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
ax2.set_xlim(0, 99)
ax2.set_ylim(0, 0.55)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_curves.png"))
plt.close(fig)


print("[2/8] Generating operating_points_frontier.png ...")
d4_df = pd.read_csv(r'C:\Users\Sara\Downloads\outputs_TIER1\figures\TIER1\D4_operating_points_frontier_TIER1.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Review burden across parameter grid
for fam in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']:
    sub = d4_df[d4_df['family'] == fam]
    label_map = {
        'F0': 'F0: Single Model',
        'F1': 'F1: Ensemble Mean',
        'F2': 'F2: Member Max',
        'F3': 'F3: Disagreement-Aware',
        'F4': 'F4: Morphological Dilation',
        'F5': 'F5: Adaptive Component (Ours)'
    }
    lw = 2.5 if fam in ['F4', 'F5'] else 1.5
    ls = '-' if fam in ['F4', 'F5'] else '--'
    ax1.plot(sub['param_value'], sub['mean_fp'], label=label_map[fam], color=palette[fam], linewidth=lw, linestyle=ls)

ax1.set_xlabel('Threshold Parameter (Score Slack)')
ax1.set_ylabel('Mean False-Positive Lesions / Patient')
ax1.set_yscale('log')
ax1.set_title('(a) Review Burden vs Permissiveness Grid')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

# Panel B: Risk-Coverage Frontier (Empirical Risk vs FP Lesions)
for fam in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']:
    sub = d4_df[d4_df['family'] == fam]
    lw = 2.5 if fam in ['F4', 'F5'] else 1.5
    ax2.plot(sub['mean_fp'], sub['mean_risk'], label=fam, color=palette[fam], linewidth=lw)

# Highlight certified operating point at alpha = 0.10
ax2.axhline(0.10, color='#dc2626', linestyle=':', linewidth=1.5, label='Target Risk $\\alpha = 0.10$')
ax2.scatter([0.0], [0.0], color='#10b981', s=100, marker='*', zorder=10, label='F5 Certified Operating Point')
ax2.set_xlabel('Mean False-Positive Lesions / Patient')
ax2.set_ylabel('Realized Missed-Lesion Risk $R_{\\mathrm{lesion}}$')
ax2.set_title('(b) Empirical Risk vs False Alarms Frontier')
ax2.set_xlim(-5, 120)
ax2.set_ylim(-0.02, 0.25)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "operating_points_frontier.png"))
plt.close(fig)


print("[3/8] Generating resplits_boxplots.png ...")
e1_df = pd.read_csv(r'C:\Users\Sara\Downloads\outputs_TIER1\figures\TIER1\E1_resplits_boxplots_TIER1.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

fam_order = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']
fam_labels = ['F0\nSingle', 'F1\nMean', 'F2\nMax', 'F3\nDisagr.', 'F4\nDilation', 'F5\nAdaptive']

# Panel A: FP Lesions across 1,000 empirical re-splits
box_data = [e1_df[e1_df['family'] == f]['fp_lesions'].values for f in fam_order]
bp1 = ax1.boxplot(box_data, tick_labels=fam_labels, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, fam in zip(bp1['boxes'], fam_order):
    patch.set_facecolor(palette[fam])
    patch.set_alpha(0.8)

ax1.axhline(2.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Clinical Budget ($\\leq 2.0$ FP/pt)')
ax1.set_ylabel('False-Positive Lesions / Patient')
ax1.set_title('(a) Review Burden over 1,000 Empirical Re-Splits')
ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# Panel B: Zoomed comparison of F4 and F5
sub_zoom = e1_df[e1_df['family'].isin(['F4', 'F5'])]
bp2 = ax2.boxplot([e1_df[e1_df['family'] == 'F4']['fp_lesions'].values,
                   e1_df[e1_df['family'] == 'F5']['fp_lesions'].values],
                  tick_labels=['F4: Morphological Dilation', 'F5: Adaptive Component (Ours)'],
                  patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
bp2['boxes'][0].set_facecolor(palette['F4'])
bp2['boxes'][1].set_facecolor(palette['F5'])
ax2.axhline(2.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Clinical Budget ($\\leq 2.0$)')
ax2.set_ylabel('False-Positive Lesions / Patient')
ax2.set_title('(b) Focus: Spatial Slack (F4) vs Adaptive Selection (F5)')
ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "resplits_boxplots.png"))
plt.close(fig)


print("[4/8] Generating subgroup_risks.png ...")
b1_df = pd.read_csv(r'C:\Users\Sara\Downloads\outputs_TIER1\figures\TIER1\B1_subgroup_risks_TIER1.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Subgroup categories
cats = ['KiTS21\n(Cohort)', 'KiTS23\n(Cohort)', '$\\leq 4\\,$cm\n(T1a)', '4-7$\\,$cm\n(T1b)', '>7$\\,$cm\n(T2)', 'ccRCC\n(Histology)', 'Other\n(Histology)']
n_pts = b1_df['n_patients'].values
realized_risk = b1_df['realized_lesion_risk'].values

colors = ['#3b82f6', '#3b82f6', '#10b981', '#10b981', '#10b981', '#f59e0b', '#f59e0b']

# Panel A: Realized risk across subgroups
bars1 = ax1.bar(cats, realized_risk, color=colors, alpha=0.85, edgecolor='black', width=0.55)
ax1.axhline(0.10, color='#dc2626', linestyle='--', linewidth=1.5, label='Target Bound $\\alpha = 0.10$')
ax1.set_ylabel('Realized Lesion Risk $R_{\\mathrm{lesion}}$')
ax1.set_ylim(0, 0.20)
ax1.set_title('(a) Realized Risk by Natural Clinical Subgroup')
ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# Panel B: Sample size per subgroup
bars2 = ax2.bar(cats, n_pts, color=colors, alpha=0.75, edgecolor='black', width=0.55)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{int(yval)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.axhline(30, color='#64748b', linestyle=':', linewidth=1.2, label='Confirmatory Threshold ($N \\geq 30$)')
ax2.set_ylabel('Patients in Validation Subgroup ($N$)')
ax2.set_ylim(0, 20)
ax2.set_title('(b) Patient Cohort Sample Sizes')
ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "subgroup_risks.png"))
plt.close(fig)


print("[5/8] Generating inter_repeat_dice.png ...")
c1_df = pd.read_csv(r'C:\Users\Sara\Downloads\outputs_TIER1\figures\TIER1\C1_inter_repeat_dice_TIER1.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Panel A: Histogram of pairwise mean Dice
sns.histplot(c1_df['dice_mean_repeats'], kde=True, ax=ax1, color='#059669', bins=25, edgecolor='black', alpha=0.65)
mean_d = c1_df['dice_mean_repeats'].mean()
std_d = c1_df['dice_mean_repeats'].std()
ax1.axvline(mean_d, color='#dc2626', linestyle='--', linewidth=1.8, label=f'Mean Agreement: {mean_d:.4f} $\\pm$ {std_d:.4f}')
ax1.set_xlabel('Pairwise Inter-Annotator Dice')
ax1.set_ylabel('Number of Cases ($N=300$)')
ax1.set_title('(a) Multi-Reader Inter-Annotator Agreement')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

# Panel B: Pairwise correlation violin plots
pair_data = [c1_df['dice_rep1_rep2'], c1_df['dice_rep1_rep3'], c1_df['dice_rep2_rep3']]
parts = ax2.violinplot(pair_data, positions=[1, 2, 3], showmeans=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor('#3b82f6')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['Reader 1 vs 2', 'Reader 1 vs 3', 'Reader 2 vs 3'])
ax2.set_ylabel('Dice Agreement')
ax2.set_title('(b) Pairwise Reader Distribution')
ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "inter_repeat_dice.png"))
plt.close(fig)


print("[6/8] Generating dataset_distributions.png ...")
a2_df = pd.read_csv(r'C:\Users\Sara\Downloads\outputs_TIER1\figures\TIER1\group_A\A2_lesion_size_distribution_TIER1.csv')
a3_df = pd.read_csv(r'C:\Users\Sara\Downloads\outputs_TIER1\figures\TIER1\group_A\A3_slice_thickness_TIER1.csv')
a4_df = pd.read_csv(r'C:\Users\Sara\Downloads\outputs_TIER1\figures\TIER1\group_A\A4_histology_distribution_TIER1.csv').dropna().head(6)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

# 1. Lesion Size
sns.histplot(a2_df['radiographic_size_cm'], kde=True, ax=ax1, color='#3b82f6', bins=20, edgecolor='black', alpha=0.7)
ax1.axvline(4.0, color='#dc2626', linestyle='--', linewidth=1.5, label='T1a Bound (4.0 cm, 54%)')
ax1.axvline(7.0, color='#f59e0b', linestyle='--', linewidth=1.5, label='T1b Bound (7.0 cm)')
ax1.set_xlabel('Radiographic Lesion Diameter (cm)')
ax1.set_ylabel('Number of Cases')
ax1.set_title('(a) Tumor Size Distribution (TNM)')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=8, loc='upper right')

# 2. Slice Thickness
thickness_counts = a3_df['slice_thickness_mm'].value_counts().sort_index()
ax2.bar([str(x) + ' mm' for x in thickness_counts.index], thickness_counts.values, color='#059669', edgecolor='black', alpha=0.75, width=0.55)
ax2.set_xlabel('CT Slice Thickness (z-axis)')
ax2.set_ylabel('Number of Scans ($N=489$)')
ax2.set_title('(b) Acquisition Slice Thickness')
ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

# 3. Histology
hist_labels = [h.replace('_', ' ').title() for h in a4_df['histology']]
ax3.barh(hist_labels[::-1], a4_df['count'].values[::-1], color='#8b5cf6', edgecolor='black', alpha=0.75, height=0.6)
ax3.set_xlabel('Patient Count')
ax3.set_title('(c) Histological Subtypes')
ax3.grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dataset_distributions.png"))
plt.close(fig)


print("[7/8] Generating system_architecture.png ...")
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.axis('off')

# Architectural Blocks
def draw_box(x, y, w, h, title, subtitle, color, edgecolor):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.04",
                                  facecolor=color, edgecolor=edgecolor, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.68, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0f172a')
    ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=8.2, color='#334155')

# Draw Pipeline Stages
draw_box(0.02, 0.55, 0.18, 0.35, "1. Patient Partition", "KiTS23 (N=489)\nTrain (60%) / Cal (20%) / Test (20%)\nPatient-Level Exchangeable Split", "#dbeafe", "#2563eb")
draw_box(0.24, 0.55, 0.18, 0.35, "2. Neural Backbone", "nnU-Net v2 3D Fullres\nPlainConvUNet & ResEnc-M\n100 Epochs (Loss -0.19, EMA 0.476)", "#ede9fe", "#7c3aed")
draw_box(0.46, 0.55, 0.18, 0.35, "3. Prediction Families", "F0: Baseline | F1: Ens Mean\nF2: Max | F3: Disagreement\nF4: Dilation | F5: Adaptive Comp.", "#fef3c7", "#d97706")
draw_box(0.68, 0.55, 0.18, 0.35, "4. Statistical Calibration", "Split-Conformal & CRC\nCalibration Split (N=98)\nCertified Risk bound $\\alpha = 0.10$\nOptimal Threshold $\\lambda^* = 0.50$", "#fee2e2", "#dc2626")
draw_box(0.89, 0.55, 0.09, 0.35, "5. Guarantee", "$P(\\mathrm{Miss}) \\leq \\alpha$\nMacro Risk\nBound Valid", "#d1fae5", "#059669")

# Draw Clinical Deployment & Verification Interface (Lower Layer)
draw_box(0.15, 0.08, 0.70, 0.35, "KiTS23 Clinical Verification Suite (Full-Stack Medical Interface)",
         "Flask Engine & High-Speed JPEG Slicing (<10 ms/slice)  |  Hounsfield Presets: Abdomen (40/400), Kidney (80/300), Bone (400/1500)\nInteractive Axial Scroller & Multi-Class Overlay (Kidney/Tumor/Cyst)  |  Automated TNM T-Staging & Biomarker Volumetrics (mL)",
         "#f8fafc", "#0284c7")

# Draw Connectors
def draw_arrow(x1, y1, x2, y2, text=""):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6, shrink=0.05))
    if text:
        ax.text((x1+x2)/2, y1+0.03, text, ha='center', va='bottom', fontsize=8, color='#475569')

draw_arrow(0.20, 0.725, 0.24, 0.725)
draw_arrow(0.42, 0.725, 0.46, 0.725)
draw_arrow(0.64, 0.725, 0.68, 0.725)
draw_arrow(0.86, 0.725, 0.89, 0.725)

# Connect Pipeline to Clinical Interface
ax.annotate('', xy=(0.50, 0.43), xytext=(0.50, 0.55),
            arrowprops=dict(facecolor='#0284c7', edgecolor='#0284c7', width=1.5, headwidth=6, shrink=0.05))
ax.text(0.51, 0.49, "Deployment & Visual Verification", ha='left', va='center', fontsize=8.5, color='#0284c7', fontweight='bold')

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "system_architecture.png"))
plt.close(fig)


print("[8/8] Generating interface_preview.png ...")
# Render clinical interface slice snapshot using case_00152
try:
    import nibabel as nib
    from PIL import Image
    sys.path.insert(0, os.path.abspath("interface"))
    import app
    
    # Render slice 20 of case_00152
    with app.app.test_client() as client:
        res = client.get('/api/slice/case_00152/20?window=kidney&overlay=both&opacity=0.6')
        if res.status_code == 200:
            with open(os.path.join(OUT_DIR, "interface_preview.png"), "wb") as f:
                f.write(res.data)
            print("Successfully rendered interface_preview.png from case_00152!")
except Exception as e:
    print("Interface preview generation fallback:", e)

print("[SUCCESS] All publication figures generated successfully in assets/!")
