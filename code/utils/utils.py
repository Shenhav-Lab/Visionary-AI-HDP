import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from scipy.stats import sem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    balanced_accuracy_score

)
import matplotlib as mpl
import re
import seaborn as sns
import textwrap


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "DejaVu Sans"

dpi = 300
font_size = 12
fig_size = 5


def roc_helper(column_start, all_data, n_points, ci_type, add_col, pw=10):
    mean_fpr = np.linspace(0, 1, n_points)

    tpr_list = []
    aucs = []

    for pw in range(1, pw + 1):
        column = f"{column_start}_{pw}"
        subset = all_data[[column, "label"] + add_col].dropna()
        probs = np.array(list(subset[[column] + add_col].mean(axis=1)))
        truths = np.array(list(subset["label"]))

        fpr, tpr, _ = roc_curve(truths, probs)
        roc_auc = roc_auc_score(truths, probs)
        aucs.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_list.append(interp_tpr)

    tpr_array = np.array(tpr_list)
    mean_tpr = np.mean(tpr_array, axis=0)
    mean_tpr[-1] = 1.0

    if ci_type == 'std':
        error = np.std(tpr_array, axis=0)
        label = '±1 std. dev.'
    elif ci_type == '95ci':
        error = sem(tpr_array, axis=0) * 1.96
        label = '95% CI'
    else:
        raise ValueError("ci_type must be 'std' or '95ci'")

    upper = np.minimum(mean_tpr + error, 1)
    lower = np.maximum(mean_tpr - error, 0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    return mean_fpr, mean_tpr, mean_auc, std_auc, lower, upper


def plot_avg_roc_from_dicts(all_data, title, color="black", n_points=100, ci_type='95ci', add_col=[], plot=True, pw=10, rand=False):
    label = ""
    mean_fpr, mean_tpr, mean_auc, std_auc, lower, upper = roc_helper("pw", all_data, n_points, ci_type, add_col, pw)
    if plot:
        plt.fill_between(mean_fpr, lower, upper, color=color, alpha=0.2, label=label)
        plt.plot(mean_fpr, mean_tpr, color=color,
                label=f'{title} (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)

        ax = plt.gca()
        if rand:
            plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label="Random (AUC = 0.5)")
        plt.xlim(0, 1)
        plt.ylim(0.00000001, 1)
        plt.xlabel('False Positive Rate (1- Specificity)', fontsize=font_size)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=font_size)
        plt.title(title)
        ax.tick_params(labelsize=font_size)
        plt.legend(loc='lower right', fontsize=(font_size-2))
        plt.grid(True, alpha=0.3)
    return mean_auc


def pr_helper(column_start, all_data, n_points, ci_type, add_col, pw=10):
    mean_recall = np.linspace(0, 1, n_points)

    precision_list = []
    aps = []

    for pw in range(1, pw):
        column = f"{column_start}_{pw}"
        subset = all_data[[column, "label"] + add_col].dropna()
        probs = np.array(list(subset[[column] + add_col].mean(axis=1)))
        truths = np.array(list(subset["label"]))

        precision, recall, _ = precision_recall_curve(truths, probs)
        ap = average_precision_score(truths, probs)
        aps.append(ap)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precision_list.append(interp_precision)

    precision_array = np.array(precision_list)
    mean_precision = np.mean(precision_array, axis=0)

    if ci_type == 'std':
        error = np.std(precision_array, axis=0)
        label = '±1 std. dev.'
    elif ci_type == '95ci':
        error = sem(precision_array, axis=0) * 1.96
        label = '95% CI'
    else:
        raise ValueError("ci_type must be 'std' or '95ci'")

    upper = np.minimum(mean_precision + error, 1)
    lower = np.maximum(mean_precision - error, 0)

    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    return mean_recall, mean_precision, mean_ap, std_ap, lower, upper


def plot_avg_pr_from_dicts(all_data, title, color="black", n_points=100, ci_type='95ci', add_col=[], plot=True, pw=10, rand=False):
    label = ""
    mean_recall, mean_precision, mean_ap, std_ap, lower, upper = pr_helper("pw", all_data, n_points, ci_type, add_col, pw)
    if plot:
        plt.fill_between(mean_recall, lower, upper, color=color, alpha=0.2, label=label)
        plt.plot(mean_recall, mean_precision, color=color,
                label=f'{title} (AP = {mean_ap:.2f} ± {std_ap:.2f})', lw=2)

        ax = plt.gca()

        plt.xlabel('Recall (Sensitivity)', fontsize=font_size)
        plt.xlim(0, 1)
        plt.ylim(0.00000001, 1)
        plt.ylabel('Precision (PPV)', fontsize=font_size)
        plt.title(title)
        # pos / (avg num pw controls + hc + pos)
        if rand:
            val = sum(list(all_data['label'])) / ((len(all_data) - len(all_data[all_data["label"] == 0].dropna()) - sum(list(all_data['label']))) / pw + len(all_data[all_data["label"] == 0].dropna()) + sum(list(all_data['label'])))
            plt.axhline(val, linestyle='--', color='grey', label=f"Random (AP = {val:.2f})")
        plt.legend(loc='lower left', fontsize=(font_size - 2))
        ax.tick_params(labelsize=font_size)
        plt.grid(True, alpha=0.3)
    return mean_ap


def get_metrics(label, prob):
    label = np.array(label)
    prob = np.array(prob)
    
    auc = roc_auc_score(label, prob)
    ap = average_precision_score(label, prob)
    
    print(f"AUC: {auc:.2f}")
    print(f"AP:    {ap:.2f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # roc
    fpr_curve, tpr_curve, _ = roc_curve(label, prob)
    axes[0].plot(fpr_curve, tpr_curve, color='blue', lw=2, label=f'AUC = {auc:.2f}')
    axes[0].plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3)
    
    # pr
    precision_curve, recall_curve, _ = precision_recall_curve(label, prob)
    axes[1].plot(recall_curve, precision_curve, color='red', lw=2, label=f'AP = {ap:.2f}')
    baseline = label.sum() / len(label)
    axes[1].axhline(y=baseline, color='grey', lw=1, linestyle='--', label=f'Baseline = {baseline:.2f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # thresholds
    thresholds = np.arange(0, 1.01, 0.01)
    results = []
    
    for thresh in thresholds:
        preds = (prob >= thresh).astype(int)
        
        tp = ((preds == 1) & (label == 1)).sum()
        fp = ((preds == 1) & (label == 0)).sum()
        tn = ((preds == 0) & (label == 0)).sum()
        fn = ((preds == 0) & (label == 1)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'tpr': tpr,
            'fpr': fpr,
            'tnr': tnr,
            'fnr': fnr,
            'ppv': ppv,
            'adjusted ppv': tpr * 0.08 / (tpr * 0.08 + (fpr) * (1 - 0.08)),
            'adjusted npv': tnr * (1 - 0.08) / (tnr * (1 - 0.08) + fnr * 0.08),
            'npv': npv,
            'f1': f1
        })
    
    df = pd.DataFrame(results)
    
    # best
    best_row = df.loc[df['f1'].idxmax()]
    print(f"\nBest threshold by F1: {best_row['threshold']:.2f} (F1={best_row['f1']:.2f})")
    
    return df


def get_test_roc(label, prob, color="black", curve_label="", rand=False):
    label = np.array(label)
    prob = np.array(prob)
    
    auc = roc_auc_score(label, prob)
    
    # roc
    fpr_curve, tpr_curve, _ = roc_curve(label, prob)
    plt.xlim(0, 1)
    plt.ylim(0.00000001, 1)
    plt.xlabel('False Positive Rate (1- Specificity)', fontsize=font_size)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=font_size)
    plt.grid(True, alpha=0.3)
    plt.plot(fpr_curve, tpr_curve, color=color, lw=2, label=f'{curve_label} (AUC = {auc:.2f})')
    if rand:
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label="Random (AUC = 0.5)")  
    plt.legend(loc='lower right', fontsize=(font_size-2))
    ax = plt.gca()
    ax.tick_params(axis='both',labelsize=font_size)
    return auc


def get_test_pr(label, prob, color="black", curve_label="", rand=False):
    label = np.array(label)
    prob = np.array(prob)
    
    ap = average_precision_score(label, prob)
    
    # roc
    fpr_curve, tpr_curve, _ = roc_curve(label, prob)
    precision_curve, recall_curve, _ = precision_recall_curve(label, prob)
    plt.plot(recall_curve, precision_curve, color=color, lw=2, label=f'{curve_label} (AP = {ap:.2f})')
    baseline = label.sum() / len(label)
    if rand:
        plt.axhline(y=baseline, color='grey', lw=1, linestyle='--', label=f'Random (AP = {baseline:.2f})')
    plt.xlabel('Recall (Sensitivity)', fontsize=font_size)
    plt.xlim(0, 1)
    plt.ylim(0.00000001, 1)
    plt.ylabel('Precision (PPV)', fontsize=font_size)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=(font_size-2))
    ax = plt.gca()
    ax.tick_params(axis='both',labelsize=font_size)
    return ap


def get_test_metrics(label, prob):
    label = np.array(label)
    prob = np.array(prob)
    
    # thresholds
    thresholds = np.arange(0, 1.01, 0.01)
    results = []
    
    for thresh in thresholds:
        preds = (prob >= thresh).astype(int)
        
        tp = ((preds == 1) & (label == 1)).sum()
        fp = ((preds == 1) & (label == 0)).sum()
        tn = ((preds == 0) & (label == 0)).sum()
        fn = ((preds == 0) & (label == 1)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'fpr': fpr,
            'tpr': tpr,
            'ppv': ppv,
            'npv': npv,
            'adjusted ppv': tpr * 0.08 / (tpr * 0.08 + (fpr) * (1 - 0.08)) if (tpr * 0.08 + (fpr) * (1 - 0.08)) > 0 else 0,
            'adjusted npv': tnr * (1 - 0.08) / (tnr * (1 - 0.08) + fnr * 0.08) if (tnr * (1 - 0.08) + fnr * 0.08) > 0 else 0,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'balanced acc': balanced_accuracy_score(label, preds)
        })
    
    df = pd.DataFrame(results)
    
    # best
    best_row = df.loc[df['f1'].idxmax()]
    print(f"\nBest threshold by F1: {best_row['threshold']:.2f} (F1={best_row['f1']:.2f})")
    
    return df


bl_clean = {
    'graph_feats_mixed_feat_set__Logistic_Regression': 'Mixed LR',
    'tda_outward_Random_Forest': 'TDA Outward RF',
    'graph_feats_ratio_feats__Logistic_Regression': 'Ratio LR',
    'graph_feats_tort_cols__Logistic_Regression': 'Tortuosity LR',
    'graph_feats_tort_cols__XGBoost': 'Tortuosity XGB',
    'tda_flooding_XGBoost': 'TDA Flooding XGB',
    'graph_feats_mixed_feat_set__Random_Forest': 'Mixed RF',
    'tda_inward_Random_Forest': 'TDA Inward RF',
    'processed_dan_tree_feats_nest_num_feats_dan__Logistic_Regression': 'Nesting Number LR',
    'tree_feats_len_bottom_XGBoost': 'Inferior Topological Length Shift XGB',
    'graph_feats_tort_cols__Random_Forest': 'Tortuosity RF',
    'graph_feats_complexity_cols__Logistic_Regression': 'Complexity LR',
    'tda_flooding_Logistic_Regression': 'TDA Flooding LR',
    'old_tree_feats_diameter_XGBoost': 'Vessel Thickness Shift XGB',
    'tree_feats_len_real_bottom_Logistic_Regression': 'Weighted Inferior Topological Length Shift LR',
    'graph_feats_topo_lengths_feats__Logistic_Regression': 'Topological Length LR',
    'graph_feats_tree_feats__Random_Forest': 'Tree RF',
    'graph_feats_topo_lengths_feats__XGBoost': 'Topological Length XGB',
    'graph_feats_complexity_cols__Random_Forest': 'Complexity RF',
    'tda_flooding_Random_Forest': 'TDA Flooding RF',
    'tda_vr_Logistic_Regression': 'TDA VR LR',
    'old_tree_feats_assymetry_Logistic_Regression': 'Asymmetry Shift LR',
    'tree_feats_len_real_top_XGBoost': 'Weighted Superior Topological Length Shift XGB',
    'tda_outward_XGBoost': 'TDA Outward XGB',
    'processed_dan_tree_feats_angle_feats_dan__XGBoost': 'Angles XGB',
    'tree_feats_len_real_bottom_Random_Forest': 'Weighted Inferior Topological Length Shift RF',
    'graph_feats_geom_feats__Logistic_Regression': 'Geometry LR',
    "graph_feats_geom_feats__XGBoost": "Geometry XGB",
    'graph_feats_mixed_feat_set__XGBoost': 'Mixed XGB',
    'tda_outward_Logistic_Regression': 'TDA Outward LR',
    'graph_feats_graph_cols__Random_Forest': 'Graph RF',
    'box_counting_df_Logistic_Regression': 'Box Counting LR',
    'graph_feats_graph_cols__XGBoost': 'Graph XGB',
    'graph_feats_complexity_cols__XGBoost': 'Complexity XGB',
    'tree_feats_conductivity_bottom_Random_Forest': 'Inferior Vessel Thickness Shift RF',
    'processed_dan_tree_feats_angle_feats_dan__Logistic_Regression': 'Angles LR',
    'processed_dan_tree_feats_angle_feats_dan__Random_Forest': 'Angles RF',
    'graph_feats_tree_feats__XGBoost': 'Tree XGB',
    'processed_dan_tree_feats_topo_lengths_feats_dan__Logistic_Regression': 'Topological Length (whole graph) LR',
    'tda_vr_Random_Forest': 'TDA VR RF',
    'tree_feats_conductivity_bottom_XGBoost': 'Inferior Vessel Thickness Shift XGB',
    'processed_dan_tree_feats_nest_num_feats_dan__Random_Forest': 'Nesting Number RF',
    'old_tree_feats_assymetry_XGBoost': 'Asymmetry Shift XGB',
    'graph_feats_tree_feats__Logistic_Regression': 'Tree LR',
    'tree_feats_conductivity_top_Random_Forest': 'Superior Vessel Thickness Shift RF',
    'graph_feats_bifurc_ratio_feats__XGBoost': 'Bifurcation Ratio XGB',
    'tda_inward_Logistic_Regression': 'TDA Inward LR',
    'graph_feats_topo_lengths_feats__Random_Forest': 'Topological Length RF',
    'tree_feats_conductivity_top_XGBoost': 'Superior Vessel Thickness Shift XGB',
    'graph_feats_bifurc_ratio_feats__Logistic_Regression': 'Bifurcation Ratio LR',
    'tree_feats_len_bottom_Random_Forest': 'Inferior Topological Length Shift RF',
    'graph_feats_ratio_feats__XGBoost': 'Ratio XGB',
    'tda_inward_XGBoost': 'TDA Inward XGB',
    'box_counting_df_XGBoost': 'Box Counting XGB',
    'tree_feats_len_bottom_Logistic_Regression': 'Inferior Topological Length Shift LR',
    'graph_feats_ratio_feats__Random_Forest': 'Ratio RF',
    'graph_feats_std_feats__XGBoost': 'STD XGB',
    'old_tree_feats_assymetry_Random_Forest': 'Asymmetry Shift RF',
    'graph_feats_cum_size_dist_feats__Logistic_Regression': 'Cum. Size Dist. LR',
    'box_counting_df_Random_Forest': 'Box Counting RF',
    'tree_feats_len_top_XGBoost': 'Superior Topological Length Shift XGB',
    'old_tree_feats_len_Random_Forest': 'Topological Length Shift RF',
    'tree_feats_len_real_top_Logistic_Regression': 'Weighted Superior Topological Length Shift LR',
    'tree_feats_conductivity_top_Logistic_Regression': 'Superior Vessel Thickness Shift LR',
    'graph_feats_conductivity_feats__Logistic_Regression': 'Vessel Thickness LR',
    'processed_dan_tree_feats_topo_lengths_feats_dan__XGBoost': 'Topological Length (whole graph) XGB',
    'graph_feats_std_feats__Logistic_Regression': 'STD LR',
    'graph_feats_asym_feats__XGBoost': 'Asymmetry XGB',
    'graph_feats_graph_cols__Logistic_Regression': 'Graph LR',
    'tree_feats_len_real_bottom_XGBoost': 'Weighted Inferior Topological Length Shift XGB',
    'graph_feats_bifurc_ratio_feats__Random_Forest': 'Bifurcation Ratio RF',
    'tree_feats_len_top_Random_Forest': 'Superior Topological Length Shift RF',
    'tree_feats_len_top_Logistic_Regression': 'Superior Topological Length Shift LR',
    'processed_dan_tree_feats_nest_num_feats_dan__XGBoost': 'Nesting Number XGB',
    'tree_feats_len_real_top_Random_Forest': 'Weighted Superior Topological Length Shift LR',
    'graph_feats_asym_feats__Logistic_Regression': 'Asymmetry LR',
    'old_tree_feats_len_Logistic_Regression': 'Topological Length Shift LR',
    'graph_feats_strahler_order_feats__Logistic_Regression': 'Strahler Order LR',
    'graph_feats_strahler_order_feats__XGBoost': 'Strahler Order XGB',
    "graph_feats_cum_size_dist_feats__XGBoost": 'Cum. Size Dist. XGB',
    "graph_feats_geom_feats__Random_Forest": 'Geometry RF',
    "graph_feats_strahler_order_feats__Random_Forest": 'Strahler Order RF',
    "tree_feats_conductivity_bottom_Logistic_Regression": 'Inferior Vessel Thickness Shift LR',
    "graph_feats_std_feats__Random_Forest":  'STD RF',
    "graph_feats_nest_num_feats__Random_Forest": "Nest Num RF",
    "graph_feats_cum_size_dist_feats__Random_Forest": 'Cum. Size Dist. RF',
    "graph_feats_nest_num_feats__Logistic_Regression": "Nest Num LR",
    "graph_feats_nest_num_feats__XGBoost": "Nest Num XGB",
    "tda_vr_XGBoost": "TDA VR XGB",
    "graph_feats_conductivity_feats__XGBoost": 'Vessel Thickness XGB',
    "processed_dan_tree_feats_topo_lengths_feats_dan__Random_Forest": 'Topological Length (whole graph) RF',
    "graph_feats_conductivity_feats__Random_Forest": 'Vessel Thickness RF',
    "graph_feats_geom_feats__XGBoost": "Geometry XGB",
    "old_tree_feats_diameter_Logistic_Regression": "Vessel Thickness Shift LR"
}

bl_to_cat = {
    'tda_outward_Random_Forest': 'Complexity',
    'tda_outward_XGBoost': 'Complexity',
    'tda_outward_Logistic_Regression': 'Complexity',
    'box_counting_df_Logistic_Regression': 'Complexity',
    'box_counting_df_Random_Forest': 'Complexity',
    'box_counting_df_XGBoost': 'Complexity',
    'tda_flooding_Logistic_Regression': 'Complexity',
    'tda_flooding_XGBoost': 'Complexity',
    'tda_flooding_Random_Forest': 'Complexity',
    'tda_vr_Logistic_Regression': 'Complexity',
    'tda_vr_Random_Forest': 'Complexity',
    'tda_vr_XGBoost': 'Complexity',
    'tda_inward_XGBoost': 'Complexity',
    'tda_inward_Logistic_Regression': 'Complexity',
    'tda_inward_Random_Forest': 'Complexity',

    'graph_feats_complexity_cols__Random_Forest': 'Mixed',
    'graph_feats_complexity_cols__XGBoost': 'Mixed',
    'graph_feats_complexity_cols__Logistic_Regression': 'Mixed',
    'graph_feats_std_feats__XGBoost': 'Mixed',
    'graph_feats_std_feats__Logistic_Regression': 'Mixed',
    "graph_feats_std_feats__Random_Forest": "Mixed",
    'graph_feats_ratio_feats__Logistic_Regression': 'Mixed',
    'graph_feats_ratio_feats__Random_Forest': 'Mixed',
    'graph_feats_ratio_feats__XGBoost': 'Mixed',
    'graph_feats_mixed_feat_set__Logistic_Regression': 'Mixed',
    'graph_feats_mixed_feat_set__XGBoost': 'Mixed',
    'graph_feats_mixed_feat_set__Random_Forest': 'Mixed',

    'graph_feats_tort_cols__Logistic_Regression': 'Geometry',
    'graph_feats_tort_cols__XGBoost': 'Geometry',
    'graph_feats_tort_cols__Random_Forest': 'Geometry',
    'tree_feats_conductivity_top_Logistic_Regression': 'Geometry',
    'tree_feats_conductivity_top_XGBoost': 'Geometry',
    'tree_feats_conductivity_top_Random_Forest': 'Geometry',
    'graph_feats_geom_feats__Logistic_Regression': 'Geometry',
    "graph_feats_geom_feats__Random_Forest": "Geometry",
    "graph_feats_geom_feats__XGBoost": "Geometry",
    'tree_feats_conductivity_bottom_Random_Forest': 'Geometry',
    'tree_feats_conductivity_bottom_Logistic_Regression': 'Geometry',
    'tree_feats_conductivity_bottom_XGBoost': 'Geometry',
    'graph_feats_conductivity_feats__Logistic_Regression': 'Geometry',
    "graph_feats_conductivity_feats__XGBoost": "Geometry",
    "graph_feats_conductivity_feats__Random_Forest": "Geometry",
    'old_tree_feats_diameter_XGBoost': 'Geometry',

    'old_tree_feats_assymetry_XGBoost': 'Nesting Tree',
    'old_tree_feats_assymetry_Logistic_Regression': 'Nesting Tree',
    'old_tree_feats_assymetry_Random_Forest': 'Nesting Tree',
    'graph_feats_asym_feats__XGBoost': 'Nesting Tree',
    'graph_feats_asym_feats__Logistic_Regression': 'Nesting Tree',
    'processed_dan_tree_feats_nest_num_feats_dan__Random_Forest': 'Nesting Tree',
    'processed_dan_tree_feats_nest_num_feats_dan__Logistic_Regression': 'Nesting Tree',
    'processed_dan_tree_feats_nest_num_feats_dan__XGBoost': 'Nesting Tree',
    'graph_feats_tree_feats__Logistic_Regression': 'Nesting Tree',
    'graph_feats_tree_feats__XGBoost': 'Nesting Tree',
    'graph_feats_tree_feats__Random_Forest': 'Nesting Tree',
    'graph_feats_nest_num_feats__Logistic_Regression': 'Nesting Tree',
    'graph_feats_nest_num_feats__XGBoost': 'Nesting Tree',
    "graph_feats_nest_num_feats__Random_Forest": "Nesting Tree",

    "processed_dan_tree_feats_topo_lengths_feats_dan__Random_Forest": "Graph",
    'processed_dan_tree_feats_topo_lengths_feats_dan__XGBoost': 'Graph',
    'processed_dan_tree_feats_topo_lengths_feats_dan__Logistic_Regression': 'Graph',
    'tree_feats_len_real_bottom_Random_Forest': 'Graph',
    'tree_feats_len_real_bottom_Logistic_Regression': 'Graph',
    'tree_feats_len_real_bottom_XGBoost': 'Graph',
    'graph_feats_topo_lengths_feats__XGBoost': 'Graph',
    'graph_feats_topo_lengths_feats__Logistic_Regression': 'Graph',
    'graph_feats_topo_lengths_feats__Random_Forest': 'Graph',
    'tree_feats_len_real_top_Logistic_Regression': 'Graph',
    'tree_feats_len_real_top_XGBoost': 'Graph',
    'tree_feats_len_real_top_Random_Forest': 'Graph',
    'graph_feats_graph_cols__XGBoost': 'Graph',
    'graph_feats_graph_cols__Logistic_Regression': 'Graph',
    'graph_feats_graph_cols__Random_Forest': 'Graph',
    'tree_feats_len_bottom_Random_Forest': 'Graph',
    'tree_feats_len_bottom_XGBoost': 'Graph',
    'tree_feats_len_bottom_Logistic_Regression': 'Graph',
    'tree_feats_len_top_Logistic_Regression': 'Graph',
    'tree_feats_len_top_XGBoost': 'Graph',
    "tree_feats_len_top_Random_Forest": "Graph",
    'old_tree_feats_len_Random_Forest': 'Graph',
    'old_tree_feats_len_Logistic_Regression': 'Graph',

    'processed_dan_tree_feats_angle_feats_dan__Logistic_Regression': 'Graph',
    'processed_dan_tree_feats_angle_feats_dan__XGBoost': 'Graph',
    'processed_dan_tree_feats_angle_feats_dan__Random_Forest': 'Graph',
    'graph_feats_bifurc_ratio_feats__XGBoost': 'Graph',
    'graph_feats_bifurc_ratio_feats__Logistic_Regression': 'Graph',
    'graph_feats_bifurc_ratio_feats__Random_Forest': 'Graph',
    'graph_feats_strahler_order_feats__Logistic_Regression': 'Graph',
    'graph_feats_strahler_order_feats__XGBoost': 'Graph',
    'graph_feats_strahler_order_feats__Random_Forest': 'Graph',


    'graph_feats_cum_size_dist_feats__Logistic_Regression': 'Nesting Tree',
    'graph_feats_cum_size_dist_feats__Random_Forest': 'Nesting Tree',
    'graph_feats_cum_size_dist_feats__XGBoost': 'Nesting Tree',
}


def wrap_labels(labels, max_width=12):
    """ wrap labels at max_width characters """
    return ['\n'.join(textwrap.wrap(label, max_width)) for label in labels]


def get_test_feat_imp(data_name, run, n_pop, model_type,
    save_path='/gpfs/data/shenhavlab/users/ca3261/Visionary_AI/src/analysis/final_figures/final_submission/main_text/fig_3/c.pdf',
    cmap="rocket_r"):

    all_oof = []
    
    all_features = set()
    all_data = []
    
    for train_pw in range(1, n_pop + 1):
        path = f"/gpfs/data/shenhavlab/PROJECTS/visionary_ai/data/columbia/{data_name}{train_pw}/{run}/meta_all_0.5_oof_bls/{model_type}/"
        name = os.listdir(path)
        path = path + name[0]
        
        with open(path, "rb") as f:
            data = pickle.load(f)
            features = list(data[3])
            all_features.update(features)
            all_data.append(data)
    
    def get_base_feature(feat_name):
        """strip trailing model type suffix like LR, RF, XGB to get the base feature name."""
        match = re.match(r'^(.*?)\s+(LR|RF|XGB)$', feat_name)
        if match:
            return match.group(1), match.group(2)
        return feat_name, ''
    
    unique_features_raw = sorted(all_features)
    unique_features_clean = pd.Series(unique_features_raw).map(bl_clean).tolist()
    
    heatmap_df = pd.DataFrame(0.0, index=unique_features_clean, columns=[f"PW{i}" for i in range(1, n_pop + 1)])
    
    for train_pw in range(1, n_pop + 1):
        data = all_data[train_pw - 1]
        features = list(data[3])
        importances = list(data[6].feature_importances_)
        
        clean_features = pd.Series(features).map(bl_clean).tolist()
        
        for clean_feat, imp in zip(clean_features, importances):
            heatmap_df.loc[clean_feat, f"PW{train_pw}"] = imp
    
    base_names = []
    model_types_list = []
    for feat in heatmap_df.index:
        base, model = get_base_feature(feat)
        base_names.append(base)
        model_types_list.append(model)
    
    heatmap_df['base_feature'] = base_names
    heatmap_df['model_type_suffix'] = model_types_list
    
    pw_cols = [f"PW{i}" for i in range(1, n_pop + 1)]
    agg_df = heatmap_df.groupby('base_feature')[pw_cols].sum()
    
    model_suffixes_per_base = heatmap_df.groupby('base_feature')['model_type_suffix'].apply(
        lambda x: sorted(set(s for s in x if s != ''))
    )
    
    display_names = {}
    for base in agg_df.index:
        suffixes = model_suffixes_per_base.get(base, [])
        if suffixes:
            display_names[base] = f"{base} ({'/'.join(suffixes)})"
        else:
            display_names[base] = base
    
    agg_df.index = [display_names[base] for base in agg_df.index]
    
    agg_df['prevalence'] = (agg_df[pw_cols] > 0).sum(axis=1)
    agg_df['mean_imp'] = agg_df[pw_cols].mean(axis=1)
    agg_df = agg_df.sort_values(['prevalence', 'mean_imp'], ascending=[False, False])
    agg_df = agg_df.drop(columns=['prevalence', 'mean_imp'])
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    cbar_kws = {'label': 'Meta-Learner Base Learner Importance'}
    
    sns.heatmap(
        agg_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        cbar_kws=cbar_kws,
        mask=(agg_df == 0),
        vmin=0,
        annot_kws={"size": 8}
    )
    
    ax.set_ylabel("Feature Set", fontsize=8)
    ax.tick_params(axis='y', labelsize=8, pad=150)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_yticklabels(wrap_labels(list(agg_df.index), 30), rotation=0, ha='left')
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(8)
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_xlabel("Control Group", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

