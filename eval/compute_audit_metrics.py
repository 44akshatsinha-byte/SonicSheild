"""
Technical Audit Analysis Script for SonicShield 200-recording benchmark.
Computes:
1. Detailed Score Distribution (Mean, Median, Std, IQR) genuine vs fake for all engines
2. Full Threshold Sweep [0.1 to 0.9] for FAR, FRR, Acc, Prec, Rec, F1
3. Detailed Speaker-level breakdown for all 38 speakers and 21 overlapping speakers
4. Precision-Recall curves chart generation
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from PIL import Image, ImageDraw, ImageFont

CSV_PATH = "results/in_the_wild/comparative_results_200.csv"
PR_CHART_PATH = "results/in_the_wild/pr_curves_comparison.png"

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} recordings from {CSV_PATH}")

engines = [
    ("Physics Engine", "score_physics"),
    ("Biology Engine", "score_biology"),
    ("AASIST Engine", "score_aasist"),
    ("Combined Baseline", "score_baseline"),
    ("Combined Hybrid", "score_hybrid")
]

# Section 1: Score Distributions
print("\n" + "="*80)
print("SECTION 1: SCORE DIRECTION & DISTRIBUTIONS (GENUINE vs FAKE)")
print("="*80)

score_stats = []
for name, col in engines:
    gen_scores = df[df["true_label"] == "bona-fide"][col]
    fake_scores = df[df["true_label"] == "spoof"][col]
    
    gen_mean, gen_med = gen_scores.mean(), gen_scores.median()
    gen_std = gen_scores.std()
    fake_mean, fake_med = fake_scores.mean(), fake_scores.median()
    fake_std = fake_scores.std()
    
    delta_mean = gen_mean - fake_mean
    delta_med = gen_med - fake_med
    
    score_stats.append({
        "Engine": name,
        "Column": col,
        "Gen_Mean": gen_mean,
        "Gen_Med": gen_med,
        "Gen_Std": gen_std,
        "Fake_Mean": fake_mean,
        "Fake_Med": fake_med,
        "Fake_Std": fake_std,
        "Delta_Mean": delta_mean,
        "Delta_Med": delta_med,
        "Separation_Dir": "Genuine > Fake" if delta_mean > 0 else "Fake > Genuine (Inverted!)"
    })
    print(f"\n{name} ({col}):")
    print(f"  Genuine (n={len(gen_scores)}): Mean={gen_mean:.4f}, Median={gen_med:.4f}, Std={gen_std:.4f}, Min={gen_scores.min():.4f}, Max={gen_scores.max():.4f}")
    print(f"  Fake    (n={len(fake_scores)}): Mean={fake_mean:.4f}, Median={fake_med:.4f}, Std={fake_std:.4f}, Min={fake_scores.min():.4f}, Max={fake_scores.max():.4f}")
    print(f"  Delta Mean (Gen - Fake): {delta_mean:+.4f} | Delta Median: {delta_med:+.4f} -> {score_stats[-1]['Separation_Dir']}")

# Section 2: Threshold Sweep across [0.1 to 0.9]
print("\n" + "="*80)
print("SECTION 2: THRESHOLD SWEEPS [0.1 to 0.9]")
print("="*80)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y_true = df["true_binary"].values # 1 = bona-fide, 0 = spoof

sweep_results = {}
for name, col in engines:
    scores = df[col].values
    sweep_results[name] = []
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    print(f"\n--- {name} (ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}) ---")
    print(f"{'Thresh':<7}{'FAR':<8}{'FRR':<8}{'Acc':<8}{'Prec':<8}{'Rec':<8}{'F1':<8}{'TP':<5}{'FP':<5}{'TN':<5}{'FN':<5}")
    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        far = fp / (tn + fp) if (tn + fp) > 0 else 0.0 # False Acceptance: Spoof classified as Genuine
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0 # False Rejection: Genuine classified as Spoof
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        sweep_results[name].append({
            "Threshold": th,
            "FAR": far,
            "FRR": frr,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        })
        print(f"{th:<7.1f}{far:<8.4f}{frr:<8.4f}{acc:<8.4f}{prec:<8.4f}{rec:<8.4f}{f1:<8.4f}{tp:<5}{fp:<5}{tn:<5}{fn:<5}")

# Section 3: Speaker-level Analysis
print("\n" + "="*80)
print("SECTION 3: SPEAKER-LEVEL ANALYSIS")
print("="*80)

speaker_counts = df.groupby(["speaker", "true_label"]).size().unstack(fill_value=0)
if "bona-fide" not in speaker_counts.columns:
    speaker_counts["bona-fide"] = 0
if "spoof" not in speaker_counts.columns:
    speaker_counts["spoof"] = 0
speaker_counts["total"] = speaker_counts["bona-fide"] + speaker_counts["spoof"]

overlapping = speaker_counts[(speaker_counts["bona-fide"] > 0) & (speaker_counts["spoof"] > 0)].copy()
disjoint_bona = speaker_counts[(speaker_counts["bona-fide"] > 0) & (speaker_counts["spoof"] == 0)].copy()
disjoint_spoof = speaker_counts[(speaker_counts["bona-fide"] == 0) & (speaker_counts["spoof"] > 0)].copy()

print(f"Total Unique Speakers in 200 Sample: {len(speaker_counts)}")
print(f"Overlapping Speakers (both bona-fide & spoof): {len(overlapping)} ({overlapping['total'].sum()} total recordings)")
print(f"Disjoint Bona-fide Only Speakers: {len(disjoint_bona)} ({disjoint_bona['total'].sum()} recordings)")
print(f"Disjoint Spoof Only Speakers: {len(disjoint_spoof)} ({disjoint_spoof['total'].sum()} recordings)")

# Detailed per-speaker accuracy on overlapping speakers
print("\n--- Overlapping Speakers Performance (Threshold = 0.5) ---")
overlapping_speakers = overlapping.index.tolist()

per_speaker_stats = []
for spk in overlapping_speakers:
    spk_df = df[df["speaker"] == spk]
    n_bona = (spk_df["true_label"] == "bona-fide").sum()
    n_spoof = (spk_df["true_label"] == "spoof").sum()
    n_tot = len(spk_df)
    
    # Calculate accuracy for AASIST, Hybrid, Baseline, Physics, Biology
    aasist_corr = ((spk_df["score_aasist"] >= 0.5) == (spk_df["true_binary"] == 1)).sum()
    hybrid_corr = ((spk_df["score_hybrid"] >= 0.5) == (spk_df["true_binary"] == 1)).sum()
    base_corr = ((spk_df["score_baseline"] >= 0.5) == (spk_df["true_binary"] == 1)).sum()
    phys_corr = ((spk_df["score_physics"] >= 0.5) == (spk_df["true_binary"] == 1)).sum()
    bio_corr = ((spk_df["score_biology"] >= 0.5) == (spk_df["true_binary"] == 1)).sum()
    
    per_speaker_stats.append({
        "Speaker": spk,
        "Bona_Fide": n_bona,
        "Spoof": n_spoof,
        "Total": n_tot,
        "AASIST_Acc": aasist_corr / n_tot,
        "Hybrid_Acc": hybrid_corr / n_tot,
        "Base_Acc": base_corr / n_tot,
        "Phys_Acc": phys_corr / n_tot,
        "Bio_Acc": bio_corr / n_tot
    })

spk_res_df = pd.DataFrame(per_speaker_stats).sort_values(by="Total", ascending=False)
print(f"{'Speaker':<25}{'Bona':<6}{'Spoof':<7}{'Total':<7}{'AASIST Acc':<12}{'Hybrid Acc':<12}{'Base Acc':<10}{'Phys Acc':<10}{'Bio Acc':<10}")
for _, r in spk_res_df.iterrows():
    print(f"{r['Speaker']:<25}{r['Bona_Fide']:<6}{r['Spoof']:<7}{r['Total']:<7}{r['AASIST_Acc']:<12.1%}{r['Hybrid_Acc']:<12.1%}{r['Base_Acc']:<10.1%}{r['Phys_Acc']:<10.1%}{r['Bio_Acc']:<10.1%}")

# Aggregate Overlapping vs Disjoint Performance
df_overlap = df[df["speaker"].isin(overlapping_speakers)]
df_disjoint = df[~df["speaker"].isin(overlapping_speakers)]

print("\n--- Overlapping vs Disjoint Subset Metrics (Threshold = 0.5) ---")
print(f"Overlapping Subset (n={len(df_overlap)}: {sum(df_overlap['true_binary']==1)} bona, {sum(df_overlap['true_binary']==0)} spoof):")
acc_aasist_ov = accuracy_score(df_overlap["true_binary"], (df_overlap["score_aasist"] >= 0.5).astype(int))
acc_hybrid_ov = accuracy_score(df_overlap["true_binary"], (df_overlap["score_hybrid"] >= 0.5).astype(int))
auc_aasist_ov = roc_auc_score(df_overlap["true_binary"], df_overlap["score_aasist"])
auc_hybrid_ov = roc_auc_score(df_overlap["true_binary"], df_overlap["score_hybrid"])
print(f"  AASIST: Accuracy={acc_aasist_ov:.4f}, ROC-AUC={auc_aasist_ov:.4f}")
print(f"  Hybrid: Accuracy={acc_hybrid_ov:.4f}, ROC-AUC={auc_hybrid_ov:.4f}")

print(f"\nDisjoint Subset (n={len(df_disjoint)}: {sum(df_disjoint['true_binary']==1)} bona, {sum(df_disjoint['true_binary']==0)} spoof):")
acc_aasist_dj = accuracy_score(df_disjoint["true_binary"], (df_disjoint["score_aasist"] >= 0.5).astype(int))
acc_hybrid_dj = accuracy_score(df_disjoint["true_binary"], (df_disjoint["score_hybrid"] >= 0.5).astype(int))
auc_aasist_dj = roc_auc_score(df_disjoint["true_binary"], df_disjoint["score_aasist"])
auc_hybrid_dj = roc_auc_score(df_disjoint["true_binary"], df_disjoint["score_hybrid"])
print(f"  AASIST: Accuracy={acc_aasist_dj:.4f}, ROC-AUC={auc_aasist_dj:.4f}")
print(f"  Hybrid: Accuracy={acc_hybrid_dj:.4f}, ROC-AUC={auc_hybrid_dj:.4f}")


# Section 4: Render Precision-Recall Curve Visualization
print("\n" + "="*80)
print("SECTION 4: GENERATING PRECISION-RECALL CURVES CHART")
print("="*80)

def render_pr_chart():
    img_w, img_h = 1000, 750
    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_sub = ImageFont.truetype("arial.ttf", 14)
        font_axis = ImageFont.truetype("arial.ttf", 13)
        font_tick = ImageFont.truetype("arial.ttf", 11)
        font_legend = ImageFont.truetype("arialbd.ttf", 12)
    except:
        font_title = font_sub = font_axis = font_tick = font_legend = ImageFont.load_default()
        
    # Title
    draw.text((img_w//2, 35), "Precision-Recall Curves Comparison (200 In-The-Wild Audio Files)", fill=(30, 41, 59), font=font_title, anchor="mm")
    draw.text((img_w//2, 60), "Evaluated on 100 Bona-Fide (Positive Class) and 100 Spoof Recordings", fill=(100, 116, 139), font=font_sub, anchor="mm")
    
    # Plot Area
    px0, py0, px1, py1 = 120, 100, 720, 640
    draw.rectangle([px0, py0, px1, py1], fill=(248, 250, 252), outline=(203, 213, 225), width=2)
    
    # Grid lines
    for i in range(11):
        x = px0 + (px1 - px0) * i / 10
        y = py0 + (py1 - py0) * i / 10
        val = i / 10.0
        # Vertical grid
        draw.line([(x, py0), (x, py1)], fill=(226, 232, 240), width=1)
        draw.text((x, py1 + 15), f"{val:.1f}", fill=(100, 116, 139), font=font_tick, anchor="mm")
        # Horizontal grid
        draw.line([(px0, y), (px1, y)], fill=(226, 232, 240), width=1)
        draw.text((px0 - 15, y), f"{1.0 - val:.1f}", fill=(100, 116, 139), font=font_tick, anchor="rm")

    # Axis Labels
    draw.text(((px0 + px1)//2, py1 + 45), "Recall (True Positive Rate for Bona-Fide Class)", fill=(30, 41, 59), font=font_axis, anchor="mm")
    draw.text((35, (py0 + py1)//2), "Precision (PPV)", fill=(30, 41, 59), font=font_axis, anchor="mm")

    # Baseline (no-skill) line for balanced positive prevalence = 0.5
    y_base = py1 - (py1 - py0) * 0.5
    draw.line([(px0, y_base), (px1, y_base)], fill=(148, 163, 184), width=2)
    draw.text((px1 - 80, y_base - 12), "No-Skill (0.50)", fill=(148, 163, 184), font=font_tick)

    # Engine configs
    configs = [
        ("Physics Engine", "score_physics", (13, 148, 136)),       # Teal
        ("Biology Engine", "score_biology", (217, 119, 6)),       # Orange/Amber
        ("AASIST Engine", "score_aasist", (225, 29, 72)),         # Rose/Red
        ("Combined Baseline", "score_baseline", (100, 116, 139)), # Slate
        ("Combined Hybrid", "score_hybrid", (37, 99, 235))        # Royal Blue
    ]

    leg_x = 750
    leg_y = 120
    draw.text((leg_x, leg_y), "MODELS & AP / PR-AUC", fill=(15, 23, 42), font=font_legend)
    leg_y += 25

    for name, col, color in configs:
        scores = df[col].values
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        
        # Plot PR curve line
        pts = []
        for r_val, p_val in zip(rec, prec):
            cx = px0 + (px1 - px0) * r_val
            cy = py1 - (py1 - py0) * p_val
            pts.append((cx, cy))
        
        # Sort pts by x ascending for clean stroke
        pts.sort(key=lambda p: p[0])
        draw.line(pts, fill=color, width=3)

        # Legend entry
        draw.line([(leg_x, leg_y + 8), (leg_x + 30, leg_y + 8)], fill=color, width=3)
        draw.text((leg_x + 40, leg_y), f"{name}", fill=(30, 41, 59), font=font_legend)
        draw.text((leg_x + 40, leg_y + 16), f"PR-AUC: {ap:.4f}", fill=(100, 116, 139), font=font_tick)
        leg_y += 42

    # Callout Box in Legend Area
    draw.rectangle([leg_x - 10, leg_y + 20, img_w - 30, leg_y + 210], fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    draw.text((leg_x, leg_y + 35), "Key Observations:", fill=(15, 23, 42), font=font_legend)
    draw.text((leg_x, leg_y + 60), "- AASIST leads with AP = 0.6677", fill=(225, 29, 72), font=font_tick)
    draw.text((leg_x, leg_y + 80), "- Hybrid AP drops to 0.5624", fill=(37, 99, 235), font=font_tick)
    draw.text((leg_x, leg_y + 100), "  due to heuristic noise dilution.", fill=(100, 116, 139), font=font_tick)
    draw.text((leg_x, leg_y + 120), "- Biology AP is 0.5619", fill=(217, 119, 6), font=font_tick)
    draw.text((leg_x, leg_y + 140), "- Physics AP is 0.5021 (~chance)", fill=(13, 148, 136), font=font_tick)
    draw.text((leg_x, leg_y + 165), "- Neither engine is probability-", fill=(100, 116, 139), font=font_tick)
    draw.text((leg_x, leg_y + 180), "  calibrated in deployment.", fill=(100, 116, 139), font=font_tick)

    img.save(PR_CHART_PATH)
    print(f"Saved Precision-Recall chart to {PR_CHART_PATH}")

render_pr_chart()
print("\nScript completed successfully.")
