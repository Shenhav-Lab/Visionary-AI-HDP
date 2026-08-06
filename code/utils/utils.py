import pandas as pd
import pickle
import os
from scipy.stats import sem
import numpy as np
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator
from scipy.stats import fisher_exact, mannwhitneyu, chi2_contingency, ttest_ind


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


def plot_avg_roc_from_dicts(all_data, title='', color="black", n_points=100, ci_type='95ci', 
                            add_col=[], plot=True, pw=10, rand=False, 
                            x_ticks=True, y_ticks=True, x_label=True, y_label=True):
    # retinal
    label = ""
    mean_fpr, mean_tpr, mean_auc, std_auc, lower, upper = roc_helper("pw", all_data, n_points, ci_type, add_col, pw)
    if plot:
        plt.fill_between(mean_fpr, lower, upper, color=color, alpha=0.2, label=label)
        if len(title)>0:
            plt.plot(mean_fpr, mean_tpr, color=color,
                    label=f'{title} (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)
        else:
            plt.plot(mean_fpr, mean_tpr, color=color,
                    label=f'AUC = {mean_auc:.2f} ± {std_auc:.2f}', lw=2)            

        ax = plt.gca()
        if rand:
            plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label="Rand (AUC = 0.5)")
        plt.xlim(0, 1)
        plt.ylim(0.00000001, 1)
        if x_label:
            plt.xlabel('FPR', fontsize=font_size) #False Positive Rate (1- Specificity)
        if y_label:
            plt.ylabel('TPR', fontsize=font_size) #True Positive Rate (Sensitivity)
        plt.title(title)
        ax.tick_params(labelsize=font_size)
        if not x_ticks:
            ax.set_xticklabels([])
        if not y_ticks:
            ax.set_yticklabels([])
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



def plot_avg_pr_from_dicts(all_data, title='', color="black", n_points=100, ci_type='95ci', 
                           add_col=[], plot=True, pw=10, rand=False,
                           x_ticks=True, y_ticks=True, x_label=True, y_label=True):
    # retinal
    label = ""
    mean_recall, mean_precision, mean_ap, std_ap, lower, upper = pr_helper("pw", all_data, n_points, ci_type, add_col, pw)
    if plot:
        plt.fill_between(mean_recall, lower, upper, color=color, alpha=0.2, label=label)
        if len(title)>0:
            plt.plot(mean_recall, mean_precision, color=color,
                    label=f'{title} (AP = {mean_ap:.2f} ± {std_ap:.2f})', lw=2)
        else:
            plt.plot(mean_recall, mean_precision, color=color,
                    label=f'AP = {mean_ap:.2f} ± {std_ap:.2f}', lw=2)

        ax = plt.gca()
        if x_label:
            plt.xlabel('Recall', fontsize=font_size) #Recall (Sensitivity)
        if y_label:
            plt.ylabel('Precision', fontsize=font_size) #Precision (PPV)

        plt.xlim(0, 1)
        plt.ylim(0.00000001, 1)

        plt.title(title)
        # pos / (avg num pw controls + hc + pos)
        if rand:
            val = sum(list(all_data['label'])) / ((len(all_data) - len(all_data[all_data["label"] == 0].dropna()) - sum(list(all_data['label']))) / pw + len(all_data[all_data["label"] == 0].dropna()) + sum(list(all_data['label'])))
            if title == "GHTN":
                plt.axhline(val, linestyle='--', color='grey', label=f"GHTN Random (AP = {val:.2f})")
            elif title == "CHTN":
                plt.axhline(val, linestyle='-', color='grey', label=f"CHTN Random (AP = {val:.2f})")
            else:
                plt.axhline(val, linestyle='-', color='grey', label=f"Rand (AP = {val:.2f})")
        plt.legend(loc='lower left', fontsize=(font_size-2))
        ax.tick_params(labelsize=font_size)
        if not x_ticks:
            ax.set_xticklabels([])
        if not y_ticks:
            ax.set_yticklabels([])
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


def get_test_roc(label, prob, color="black", curve_label="", rand=False, 
                            x_ticks=True, y_ticks=True, x_label=True, y_label=True):
    label = np.array(label)
    prob = np.array(prob)

    auroc = roc_auc_score(label, prob)

    # roc
    fpr_curve, tpr_curve, _ = roc_curve(label, prob)
    plt.xlim(0, 1)
    plt.ylim(0.00000001, 1)
    if x_label:
        plt.xlabel('FPR', fontsize=font_size)#False Positive Rate (1- Specificity)
    if y_label:
        plt.ylabel('TPR', fontsize=font_size)#True Positive Rate (Sensitivity)
    plt.grid(True, alpha=0.3)
    if len(curve_label)>0:
        plt.plot(fpr_curve, tpr_curve, color=color, lw=2, label=f'{curve_label} (AUC = {auroc:.2f})')
    else:
        plt.plot(fpr_curve, tpr_curve, color=color, lw=2, label=f'AUC = {auroc:.2f}')
    if rand:
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label="Rand (AUC = 0.5)")
    ax = plt.gca()
    ax.tick_params(axis='both',labelsize=font_size)
    if not x_ticks:
        ax.set_xticklabels([])
    if not y_ticks:
        ax.set_yticklabels([])
    plt.legend(loc='lower right', fontsize=(font_size-2))
    return auroc



def get_test_pr(label, prob, color="black", curve_label="", rand=False,
                x_ticks=True, y_ticks=True, x_label=True, y_label=True):
    label = np.array(label)
    prob = np.array(prob)

    ap = average_precision_score(label, prob)

    # roc
    fpr_curve, tpr_curve, _ = roc_curve(label, prob)
    precision_curve, recall_curve, _ = precision_recall_curve(label, prob)
    if len(curve_label)>0:
        plt.plot(recall_curve, precision_curve, color=color, lw=2, label=f'{curve_label} (AP = {ap:.2f})')
    else:
        plt.plot(recall_curve, precision_curve, color=color, lw=2, label=f'AP = {ap:.2f}')
    baseline = label.sum() / len(label)
    if rand:
        plt.axhline(y=baseline, color='grey', lw=1, linestyle='--', label=f'Rand (AP = {baseline:.2f})')
    if x_label:
        plt.xlabel('Recall', fontsize=font_size) #Recall (Sensitivity)
    if y_label:
        plt.ylabel('Precision', fontsize=font_size) #Precision (PPV)
    plt.xlim(0, 1)
    plt.ylim(0.00000001, 1)

    ax = plt.gca()
    ax.tick_params(axis='both',labelsize=font_size)
    if not x_ticks:
        ax.set_xticklabels([])
    if not y_ticks:
        ax.set_yticklabels([])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=(font_size-2))
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


def average_by_subject(df, ids_to_use=None,filename_to_use=None, id_regex=r"(cu\d{4})", average=True):
    """
    If a subject has both eyes, we average their features.

    Args:
        df: df of data
        ids_to_use: ids to apply this to
        filename_to_use: filenames apply this to
        id_regex: the format of the ids
        average: if we ant to combine and average them
    
    Returns:
        df with features averaged by subject.
    """
    df = df.copy()
    if 'id' in df.columns:
        pass
    elif 'file_name' in df.columns:
        df['id'] = df['file_name'].str.extract(id_regex)[0]
    elif 'index' in df.columns:
        df = df.rename(columns={'index': 'file_name'})
        df['id'] = df['file_name'].str.extract(id_regex)[0]
    if ids_to_use is not None:
        df = df[df['id'].isin(ids_to_use)]
    elif filename_to_use is not None:
        if 'file_name' in df.columns:
            df = df[df['file_name'].isin(filename_to_use)]

    if average:

        agg = {c: ('mean' if np.issubdtype(dt, np.number) else 'first')
            for c, dt in df.dtypes.items() if c != 'file_name'}
        g = df.groupby('id', as_index=False).agg(agg)
        g['file_name'] = g['id']
        return g
    else:
        return(df)



def get_pca_df(chosen_df, n_comps=2, get_pca=False, subset_fit=[]):
    if len(subset_fit)>0:
        chosen_df_fit = chosen_df[chosen_df['id'].isin(subset_fit)]
        X_fit = chosen_df_fit.drop(['file_name', 'label', 'id', 'lat'], axis=1, errors='ignore').copy()
        X_fit = X_fit.apply(pd.to_numeric, errors='coerce')        
        X_fit.replace([np.inf, -np.inf], np.nan, inplace=True) 
        X_fit = X_fit.fillna(0)                                    
        X_fit = X_fit.where(X_fit >= 1e-6, 0)
        scaler = StandardScaler()
        chosen_df_scaled_fit = scaler.fit_transform(X_fit)
        n = max(1, min(n_comps, chosen_df_scaled_fit.shape[0], chosen_df_scaled_fit.shape[1]))
        pca = PCA(n_components=n, random_state=0)
        pca.fit(chosen_df_scaled_fit)


    X = chosen_df.drop(['file_name', 'label', 'id', 'lat'], axis=1, errors='ignore').copy()

    # make sure everything going into scaler/PCA is numeric and clean
    X = X.apply(pd.to_numeric, errors='coerce')        
    X.replace([np.inf, -np.inf], np.nan, inplace=True) 
    X = X.fillna(0)                                    
    X = X.where(X >= 1e-6, 0)

    if len(subset_fit)>0:
        chosen_df_scaled = scaler.transform(X)
        chosen_df_scaled = pca.transform(chosen_df_scaled)
    else:
        scaler = StandardScaler()
        chosen_df_scaled = scaler.fit_transform(X)

        # cap n_comps to valid range
        n = max(1, min(n_comps, chosen_df_scaled.shape[0], chosen_df_scaled.shape[1]))
        pca = PCA(n_components=n, random_state=0)
        chosen_df_scaled = pca.fit_transform(chosen_df_scaled)

    # keep original structure/variable names
    chosen_df_scaled = pd.DataFrame(chosen_df_scaled, index=chosen_df.index)
    chosen_df_scaled['id'] = chosen_df['id'].tolist() if 'id' in chosen_df.columns else None
    chosen_df_scaled['file_name'] = chosen_df['file_name'].tolist() if 'file_name' in chosen_df.columns else None
    chosen_df_scaled['label'] = chosen_df['label'].tolist() if 'label' in chosen_df.columns else None

    chosen_df = chosen_df_scaled
    if get_pca:
        return chosen_df, pca
    else:
        return chosen_df

# remove outliers outside of 4*IQR
def remove_outliers_iqr(df, cols=[], iqr_mult=4):
    cleaned = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_mult * IQR
        upper = Q3 + iqr_mult * IQR
        cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
        outliers = df[(df[col] < lower) | (df[col] > upper)]['id'].tolist()
    return cleaned, outliers


def box_plot(input_df, name_map, pw_ids,pec_ids,clean_controls, get_mean=False, file_name=''):
    df = input_df.copy()
    df = df[df['id'].isin(pec_ids + pw_ids)]  # keep relevant subjects


    # Build category mapping (each subject can belong to multiple groups)
    category_map = {
        'PEC': set(pec_ids),
        'HC': set(clean_controls),
        'PW': set(pw_ids),
    }

    # Expand df so that subjects with multiple memberships get duplicated
    df_expanded = []
    for _, row in df.iterrows():
        subject_id = row['id']
        for label, id_set in category_map.items():
            if subject_id in id_set:
                new_row = row.copy()
                new_row['label'] = label
                df_expanded.append(new_row)
    df = pd.DataFrame(df_expanded)

    palette = {'PEC':'#FF0044', 'HC': 'whitesmoke', 'PW':'lightgrey'}

    xval = 'label'
    yval = list(name_map.keys())[0]

    if get_mean:
        if f'{yval}' not in df.columns:
            df[f'{yval}'] = df.filter(like='cu').mean(axis=1)

    fig, ax = plt.subplots(figsize = (2.5, 4), constrained_layout=True)

    df_ = remove_outliers_iqr(df, [yval])[0]
    df_[xval] = df_[xval].replace({0: 'HC', 1:'PEC', 2:'PW'})#, 3:'EOPE', 4:'LOPE'
    sns.boxplot(df_, x=xval, y=yval, showfliers=False, saturation=0.9, palette=palette,hue=xval, order = ['HC','PW', 'PEC'], linecolor="black")
    sns.stripplot(df_, x=xval, y=yval,color='black' , marker="$\\circ$", alpha = 0.2, edgecolor='k', linewidth=0.6, order = ['HC','PW', 'PEC'], jitter=0.2, size=4)

    annotator = Annotator(ax = ax, data = df_, x = xval, y = yval, pairs = [('HC','PEC'), ('PW','PEC')], order = ['HC','PW', 'PEC'])
    annotator.hide_non_significant=True
    annotator.configure(test="Mann-Whitney", verbose=False,line_height=0.02, text_offset=-2,text_format='star',use_fixed_offset=10)

    annotator.apply_test()

    annotator.annotate(line_offset_to_group=0.01)

    ax.tick_params(axis='y', which='major', labelsize=9)
    ax.set_xlabel('')
    ax.set_ylabel(name_map[yval])
    if len(file_name)>0:
        fig.savefig(f'figures/{file_name}.png', transparent=True, dpi=300, bbox_inches='tight')   


def box_plot_hdp(input_df, name_map, pw_ids,ght_cases,cht_cases,clean_controls, get_mean=False, file_name=''):
    df = input_df.copy()
    df = df[df['id'].isin(ght_cases+cht_cases + pw_ids)]  # keep relevant subjects

    # Build category mapping (each subject can belong to multiple groups)
    category_map = {
        'GHTN': set(ght_cases),
        'CHTN': set(cht_cases),
        'HC': set(clean_controls),
        'PW': set(pw_ids)
    }

    # Expand df so that subjects with multiple memberships get duplicated
    df_expanded = []
    for _, row in df.iterrows():
        subject_id = row['id']
        for label, id_set in category_map.items():
            if subject_id in id_set:
                new_row = row.copy()
                new_row['label'] = label
                df_expanded.append(new_row)
    df = pd.DataFrame(df_expanded)

    palette = {'CHTN':'darkorchid','GHTN':'cornflowerblue', 'HC': 'whitesmoke', 'PW':'lightgrey'}

    xval = 'label'
    yval = list(name_map.keys())[0]

    if get_mean:
        if f'{yval}' not in df.columns:
            df[f'{yval}'] = df.filter(like='cu').mean(axis=1)

    fig, ax = plt.subplots(figsize = (4, 4), constrained_layout=True)

    df_ = remove_outliers_iqr(df, [yval])[0]
    df_[xval] = df_[xval].replace({0: 'HC', 1:'PW', 2:'CHTN', 3:'GHTN'})
    sns.boxplot(df_, x=xval, y=yval, showfliers=False, saturation=0.9, palette=palette,hue=xval, order = ['HC','PW','GHTN','CHTN'], linecolor="black")#,'EOPE', 'LOPE'
    sns.stripplot(df_, x=xval, y=yval,color='black' , marker="$\\circ$", alpha = 0.2, edgecolor='k', linewidth=0.6, order = ['HC','PW','GHTN','CHTN'], jitter=0.2, size=4)
    annotator = Annotator(ax = ax, data = df_, x = xval, y = yval, pairs = [('HC','GHTN'), ('PW','GHTN'),('HC','CHTN'), ('PW','CHTN')], order = ['HC','PW','GHTN','CHTN'])
    annotator.hide_non_significant=True
    annotator.configure(test="Mann-Whitney", verbose=False,line_height=0.02, text_offset=-2,text_format='star',use_fixed_offset=10)

    annotator.apply_test()

    annotator.annotate(line_offset_to_group=0.01)

    ax.tick_params(axis='y', which='major', labelsize=9)
    ax.set_xlabel('')
    ax.set_ylabel(name_map[yval])

    if len(file_name)>0:
        fig.savefig(f'figures/{file_name}.pdf', transparent=True, dpi=300, bbox_inches='tight')   


def prepare_feature_catalog(
    case_file, control_file, clinical_file,
    id_regex=r"(cu\d{4})", average=True, feature_folder="all_features", pre="all", post=""
):
    """
    Loads all tables, builds subject-level aggregates, merges clinical/bins, defines feature groups:
      - dfs, ks_dfs, pca_dfs (lists of feature *names*)
      - dfs_map, ks_dfs_map, pca_dfs_map (name -> DataFrame)
      - dfs_set, ks_dfs_set, pca_dfs_set (sets of names)
      - df_use (with labels)
    """

    # base labels/subject filtering 
    df_results_all = pd.read_csv(os.path.join(feature_folder, f"{pre}_graph_feats_processed{post}.csv"))
    df_results_all['id'] = df_results_all['id'].astype(str).str.lower()


    with open(case_file, "rb") as f:
        case_group = pickle.load(f)

    with open(control_file, "rb") as f:
        control_group = pickle.load(f)

    case_group = [x.lower() for x in case_group]
    control_group = [x.lower() for x in control_group]

    df_use = df_results_all.copy()
    df_use['label'] = df_use['id'].apply(lambda x: 1 if x in case_group else 0)
    df_use = df_use.dropna()

    # load features
    graph_feats = df_use.copy()

    bins_feats = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_old_processed{post}.csv"))
    bins_feats['id'] = bins_feats['file_name'].str.extract(id_regex)[0]

    clinical_feats = pd.read_csv(clinical_file, index_col=0)
    clinical_feats['Study ID'] = clinical_feats['Study ID'].str.lower()

    box_counting_df = pd.read_csv(os.path.join(feature_folder, f"{pre}_box_counting{post}.csv"))
    tda_vr          = pd.read_csv(os.path.join(feature_folder, f"{pre}_tda_VR{post}.csv"))
    tda_inward      = pd.read_csv(os.path.join(feature_folder, f"{pre}_tda_inward{post}.csv"))
    tda_outward     = pd.read_csv(os.path.join(feature_folder, f"{pre}_tda_outward{post}.csv"))
    tda_flooding    = pd.read_csv(os.path.join(feature_folder, f"{pre}_tda_flooding{post}.csv"))

    tree_feats_len_top        = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_lengths_top{post}.csv"))
    tree_feats_len_real_top   = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_lengths_real_top{post}.csv"))
    tree_feats_conductivity_top = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_conductivity_top{post}.csv"))
    tree_feats_len_bottom     = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_lengths_bottom{post}.csv"))
    tree_feats_len_real_bottom = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_lengths_real_bottom{post}.csv"))
    tree_feats_conductivity_bottom = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_conductivity_bottom{post}.csv"))

    old_tree_feats_len       = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_old_lengths{post}.csv"))
    old_tree_feats_assymetry = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_old_Asymmetry{post}.csv"))
    old_tree_feats_diameter  = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_old_diameters{post}.csv"))

    processed_dan_tree_feats = pd.read_csv(os.path.join(feature_folder, f"{pre}_tree_feats_processed{post}.csv")) 


    graph_feats = graph_feats.merge(processed_dan_tree_feats, left_on='file_name', right_on='index', how='outer')
    bins_feats = graph_feats.merge(bins_feats, on='file_name', suffixes=['dan',''])

    clinical_feats['id'] = clinical_feats['Study ID'].str.lower()
    clinical_feats['file_name'] = clinical_feats['id']
    clinical_feats['label'] = [1 if x in case_group else 0 for x in clinical_feats['id']]

    # column groups 
    required = ['file_name', 'label', 'id']

    bifurc_ratio_feats    = [f'bifurc_ratio_{n}' for n in range(2, 4)]
    strahler_order_feats  = [f'strahler_order_counts_{n}' for n in range(1, 6)]
    graph_cols            = ['num_nodes','num_components','n_edge_nodes','n_branches','direct_dist']
    tort_cols             = ['sinuosity_all','dic_tort','tort_density','linreg_tort','sq_curvature_tortuosity','abs_curvature_tortuosity']
    complexity_cols       = ['loops','fds']
    asym_feats            = ['avg_asymmetry','sum_asymmetry','q25_asymmetry','q50_asymmetry','q75_asymmetry','q100_asymmetry']
    cum_size_dist_feats   = ['avg_cumulative_size_dist','sum_cumulative_size_dist','q25_cumulative_size_dist','q50_cumulative_size_dist','q75_cumulative_size_dist','q100_cumulative_size_dist']
    topo_lengths_feats    = ['avg_topo_length_top','avg_topo_length_bottom','median_topo_length_top','median_topo_length_bottom',
                             'avg_topo_length_weight_top','avg_topo_length_weight_bottom','median_topo_length_weight_top','median_topo_length_weight_bottom']
    topo_lengths_feats_dan= ['avg_topo_length_top_dan','avg_topo_length_bottom_dan','median_topo_length_top_dan','median_topo_length_bottom_dan',
                             'avg_topo_length_weight_top_dan','avg_topo_length_weight_bottom_dan','median_topo_length_weight_top_dan','median_topo_length_weight_bottom_dan']
    nest_num_feats        = ['nest_num_unw','nest_num_w','avg_nest_ratios','avg_nest_ratios_w','sum_nest_ratios','sum_nest_ratios_w']
    nest_num_feats_dan    = ['nest_num_unw_dan','nest_num_w_dan','avg_nest_ratios_dan','avg_nest_ratios_w_dan','sum_nest_ratios_dan','sum_nest_ratios_w_dan']
    tree_feats      = ['tree_depth','tree_leaves']
    geom_feats            = ['vein_density_len','mean_vein_distances','mean_areole_area','areole_density']
    angle_feats           = ['avg_angle','median_angle','sum_angle','beta']
    angle_feats_dan       = ['avg_angle_dan','median_angle_dan','sum_angle_dan']
    conductivity_feats    = ['avg_conductivities_top','median_conductivities_top','avg_conductivities_bottom','median_conductivities_bottom']
    ratio_feats           = ['avg_topo_length_weight_topbottom_ratio','avg_topo_length_topbottom_ratio',
                             'avg_conductivities_topbottom_ratio','median_conductivities_topbottom_ratio',
                             'median_topo_length_weight_topbottom_ratio','median_topo_length_topbottom_ratio']
    std_feats             = ['std_asymmetry','std_cumulative_size_dist','std_topo_length_top','std_topo_length_bottom',
                             'std_topo_length_weight_top','std_topo_length_weight_bottom','std_conductivities_top','std_conductivities_bottom',
                             'std_nest_ratios','std_nest_ratios_w','std_angle']
    clinical_feats_base   = ['Age at enrollment','Current Smoker','Former Smoker','Past Preeclampsia','IVF','Maternal hispanic',
                             'Maternal Race_Asian','Maternal Race_Black','Maternal Race_Multiracial','Maternal Race_Other',
                             'Maternal Race_Unknown','Maternal Race_White']
    clinical_feats_extra2 = ['ama_custom','Past Gestational HTN','Past Gestational diabetes','Past Hypertension']
    clinical_feats_extra = ['ama', 'gestationaldiabetes', 'obesity', 'aspirin', 'bp_meds', 'insulin']

    all_clinical_feats = ['Age at enrollment','Current Smoker','Former Smoker','Past Preeclampsia','Past Gestational HTN','Past Gestational diabetes','Past Hypertension',
                          'IVF','Maternal hispanic','Maternal Race_Asian','Maternal Race_Black','Maternal Race_Multiracial','Maternal Race_Other',
                             'Maternal Race_Unknown','Maternal Race_White', 'obesity']
    
    fmf_clinical_feats = ['Age at enrollment', 'Current Smoker', 'Past Preeclampsia',
       'IVF', 'Insulin?', 'Maternal hispanic',
       'Maternal Race_Asian', 'Maternal Race_Black',
       'Maternal Race_Multiracial', 'Maternal Race_Other',
       'Maternal Race_Unknown', 'Maternal Race_White',
       'Cardiac Hypertensive Disease', 'Diabetes_type1', 'Diabetes_type2',
       'Past Gestational diabetes', 'Past Preterm Labor', 'nulliparous']
    
    mixed_feat_set = ['n_branches','direct_dist','sq_curvature_tortuosity','loops','fds','avg_topo_length_top_dan','avg_topo_length_bottom_dan']

    # append required
    def add_req(lst): return lst + required
    graph_cols              = add_req(graph_cols)
    tort_cols               = add_req(tort_cols)
    complexity_cols         = add_req(complexity_cols)
    asym_feats              = add_req(asym_feats)
    cum_size_dist_feats     = add_req(cum_size_dist_feats)
    bifurc_ratio_feats      = add_req(bifurc_ratio_feats)
    strahler_order_feats    = add_req(strahler_order_feats)
    topo_lengths_feats      = add_req(topo_lengths_feats)
    topo_lengths_feats_dan  = add_req(topo_lengths_feats_dan)
    nest_num_feats          = add_req(nest_num_feats)
    nest_num_feats_dan      = add_req(nest_num_feats_dan)
    tree_feats        = add_req(tree_feats)
    geom_feats              = add_req(geom_feats)
    angle_feats             = add_req(angle_feats)
    angle_feats_dan         = add_req(angle_feats_dan)
    conductivity_feats      = add_req(conductivity_feats)
    ratio_feats             = add_req(ratio_feats)
    clinical_feats_base     = add_req(clinical_feats_base)
    clinical_feats_extra2   = add_req(clinical_feats_extra2)
    clinical_feats_extra   = add_req(clinical_feats_extra)
    all_clinical_feats   = add_req(all_clinical_feats)
    fmf_clinical_feats   = add_req(fmf_clinical_feats)
    std_feats               = add_req(std_feats)
    mixed_feat_set = add_req(mixed_feat_set)

    # subject-level aggregation (average)
    file_name_to_use = set(df_use['file_name'].unique())
    ids_to_use = set(df_use['id'].unique())

    box_counting_df = average_by_subject(box_counting_df, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tda_vr          = average_by_subject(tda_vr, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tda_inward      = average_by_subject(tda_inward, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tda_outward     = average_by_subject(tda_outward, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tda_flooding    = average_by_subject(tda_flooding, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    graph_feats     = average_by_subject(graph_feats, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    old_tree_feats_len       = average_by_subject(old_tree_feats_len, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    old_tree_feats_assymetry = average_by_subject(old_tree_feats_assymetry, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    old_tree_feats_diameter  = average_by_subject(old_tree_feats_diameter, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tree_feats_len_top       = average_by_subject(tree_feats_len_top, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tree_feats_len_real_top  = average_by_subject(tree_feats_len_real_top, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tree_feats_conductivity_top = average_by_subject(tree_feats_conductivity_top, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tree_feats_len_bottom    = average_by_subject(tree_feats_len_bottom, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tree_feats_len_real_bottom = average_by_subject(tree_feats_len_real_bottom, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    tree_feats_conductivity_bottom = average_by_subject(tree_feats_conductivity_bottom, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    processed_dan_tree_feats = average_by_subject(processed_dan_tree_feats, filename_to_use=file_name_to_use, id_regex=id_regex, average=average)
    processed_dan_tree_feats['label'] = processed_dan_tree_feats['id'].apply(lambda x: 1 if x in case_group else 0)   


    dfs = [
        'graph_feats[mixed_feat_set]','graph_feats[graph_cols]','graph_feats[tort_cols]','graph_feats[complexity_cols]',
        'bins_feats[asym_feats]','bins_feats[cum_size_dist_feats]','bins_feats[bifurc_ratio_feats]',
        'bins_feats[strahler_order_feats]','bins_feats[topo_lengths_feats]',
        'processed_dan_tree_feats[topo_lengths_feats_dan]','bins_feats[nest_num_feats]',
        'processed_dan_tree_feats[nest_num_feats_dan]','bins_feats[tree_feats]',
        'bins_feats[geom_feats]',
        'processed_dan_tree_feats[angle_feats_dan]','bins_feats[conductivity_feats]',
        'bins_feats[ratio_feats]','bins_feats[std_feats]',
        'clinical_feats[all_clinical_feats]'
    ]
    ks_dfs  = ['tree_feats_len_top','tree_feats_len_real_top','tree_feats_conductivity_top',
               'tree_feats_len_bottom','tree_feats_len_real_bottom','tree_feats_conductivity_bottom',
               'old_tree_feats_len','old_tree_feats_assymetry','old_tree_feats_diameter']
    pca_dfs = ['box_counting_df','tda_vr','tda_inward','tda_outward','tda_flooding']

    # dataframes by base name
    frames = {
        'graph_feats': graph_feats,
        'processed_dan_tree_feats': processed_dan_tree_feats,
        'bins_feats': bins_feats,
        'box_counting_df': box_counting_df,
        'tda_vr': tda_vr,
        'tda_inward': tda_inward,
        'tda_outward': tda_outward,
        'tda_flooding': tda_flooding,
        'tree_feats_len_top': tree_feats_len_top,
        'tree_feats_len_real_top': tree_feats_len_real_top,
        'tree_feats_conductivity_top': tree_feats_conductivity_top,
        'tree_feats_len_bottom': tree_feats_len_bottom,
        'tree_feats_len_real_bottom': tree_feats_len_real_bottom,
        'tree_feats_conductivity_bottom': tree_feats_conductivity_bottom,
        'old_tree_feats_len': old_tree_feats_len,
        'old_tree_feats_assymetry': old_tree_feats_assymetry,
        'old_tree_feats_diameter': old_tree_feats_diameter,
        'clinical_feats':clinical_feats
    }
    # column groups by feature set name
    colsets = {
        'mixed_feat_set':mixed_feat_set,'graph_cols': graph_cols, 'tort_cols': tort_cols, 'complexity_cols': complexity_cols,
        'asym_feats': asym_feats, 'cum_size_dist_feats': cum_size_dist_feats,
        'bifurc_ratio_feats': bifurc_ratio_feats, 'strahler_order_feats': strahler_order_feats,
        'topo_lengths_feats': topo_lengths_feats, 'topo_lengths_feats_dan': topo_lengths_feats_dan,
        'nest_num_feats': nest_num_feats, 'nest_num_feats_dan': nest_num_feats_dan,
        'tree_feats': tree_feats, 'geom_feats': geom_feats,
        'angle_feats': angle_feats, 'angle_feats_dan': angle_feats_dan,
        'conductivity_feats': conductivity_feats, 'ratio_feats': ratio_feats,
        'std_feats': std_feats,
        'all_clinical_feats': all_clinical_feats
    }

    # Build maps consistent with string keys
    dfs_map = {}
    for name in dfs:
        if '[' in name:        # e.g., 'graph_feats[graph_cols]'
            base, cols = name.split('[', 1)
            base = base.strip()
            cols = cols.rstrip(']').strip()
            dfs_map[name] = frames[base][colsets[cols]].copy()
        else:
            dfs_map[name] = frames[name].copy()

    ks_dfs_map  = {name: frames[name].copy() for name in ks_dfs}
    pca_dfs_map = {name: frames[name].copy() for name in pca_dfs}

    df_use = df_use[['id','label']].drop_duplicates()

    return {
        "df_use": df_use,
        "dfs": dfs, "ks_dfs": ks_dfs, "pca_dfs": pca_dfs,
        "dfs_map": dfs_map, "ks_dfs_map": ks_dfs_map, "pca_dfs_map": pca_dfs_map,
        "dfs_set": set(dfs), "ks_dfs_set": set(ks_dfs), "pca_dfs_set": set(pca_dfs),
    }


def create_disease_binary_columns(df, disease_column):
    """
    create binary column in metadata from column that hold list of diseases per subject
    """
    df = df.copy()
    # Ensure the disease column is processed as a list if it's a string
    if df[disease_column].dtype == 'object':
        # Split string if diseases are comma-separated
        df[disease_column] = df[disease_column].str.replace(' ', '', regex=False).str.split(',').apply(
            lambda x: [i.strip() for i in x] if isinstance(x, list) else []
        )
    
    # Get unique diseases across all rows
    all_diseases = set()
    for diseases in df[disease_column]:
        if isinstance(diseases, list):
            all_diseases.update(diseases)
    
    # Create binary columns for each disease
    for disease in sorted(all_diseases):
        column_name = f'{disease.lower().replace(" ", "_")}'
        df[disease_column+'_'+column_name] = df[disease_column].apply(
            lambda x: 1 if disease in x else 0
        )
    
    return df


def prepare_clinical_file(md):
    md['Study ID'] = md['Study ID'].str.lower()
    # fill it in if there is a date
    md.loc[md['Date of Hypertension Diagnosis'].notna() & md['Diagnoses, Maternal, Hypertension'].isna(), 'Diagnoses, Maternal, Hypertension'] = 'temp_hypertension'
    md.loc[md['Diagnoses, Maternal, Hypertension, Preeclampsia'].notna() & md['Diagnoses, Maternal, Hypertension'].isna(), 'Diagnoses, Maternal, Hypertension'] = 'temp_hypertension'


    # create df with binary disease columns
    md_df = md[['Study ID','Medical Diagnoses','Cardiac Disease','Has the patient had any eye surgery in the past five years?','Diagnoses, Maternal','Diagnoses, Maternal, Hypertension','Diagnoses, Maternal, Gestational Diabetes',
                'Aspirin?','Statin?','BP meds?','Insulin?','Diagnoses, Maternal, Gestational Diabetes, A2','Tobacco Use']]

    md_df = md_df.rename(columns={'Medical Diagnoses':'med_diag','Cardiac Disease':'card','Has the patient had any eye surgery in the past five years?':'eye_surg','Diagnoses, Maternal':'diag_mat',
                        'Diagnoses, Maternal, Hypertension':'diag_mat_ht', 'Diagnoses, Maternal, Gestational Diabetes':'diag_mat_gd','Diagnoses, Maternal, Gestational Diabetes, A2':'diag_mat_gd_a2'})

    md_df = create_disease_binary_columns(md_df,'med_diag').drop('med_diag', axis=1).drop('med_diag_nonereported',axis=1)
    md_df = create_disease_binary_columns(md_df,'card').drop('card', axis=1)
    md_df = create_disease_binary_columns(md_df,'eye_surg').drop('eye_surg',axis=1).drop('eye_surg_none', axis=1)
    md_df = create_disease_binary_columns(md_df,'diag_mat').drop('diag_mat',axis=1)
    md_df = create_disease_binary_columns(md_df,'diag_mat_ht').drop('diag_mat_ht',axis=1)
    md_df = create_disease_binary_columns(md_df,'diag_mat_gd').drop('diag_mat_gd',axis=1)
    md_df = create_disease_binary_columns(md_df,'Aspirin?').drop('Aspirin?',axis=1).drop('Aspirin?_no',axis=1)
    md_df = create_disease_binary_columns(md_df,'Statin?').drop('Statin?',axis=1).drop('Statin?_no',axis=1)
    md_df = create_disease_binary_columns(md_df,'Insulin?').drop('Insulin?',axis=1).drop('Insulin?_no',axis=1)
    md_df = create_disease_binary_columns(md_df,'BP meds?').drop('BP meds?',axis=1).drop('BP meds?_no',axis=1)
    md_df = create_disease_binary_columns(md_df,'Tobacco Use').drop('Tobacco Use',axis=1)

    md_select = md.merge(md_df,on='Study ID')[['Study ID','Age at enrollment','Tobacco Use_currentsmoker','Tobacco Use_formersmoker','Past Pregnancies Diagnoses, Maternal, Hypertension','diag_mat_ivf','Maternal Ethnicity','Maternal Race',
    'Cardiac Disease','diag_mat_ama','med_diag_obesity','Diabetes Mellitus','Past Pregnancies Diagnoses, Maternal, Hypertension','Past Pregnancies Diagnoses, Maternal']]

    md_select = md_select.rename(columns={'Tobacco Use_currentsmoker':'Current Smoker','Tobacco Use_formersmoker':'Former Smoker',
                                        'diag_mat_ivf':'IVF','diag_mat_ama':'ama','med_diag_obesity':'obesity'})
    md_select = md_select.loc[:, ~md_select.columns.duplicated()]
    md_select['Past Preeclampsia'] = md_select['Past Pregnancies Diagnoses, Maternal, Hypertension'].apply(lambda x: 1 if x=='Preeclampsia' else 0)
    md_select['Maternal hispanic'] = md_select['Maternal Ethnicity'].map({
        'Hispanic or Latinx': 1,
        'NOT Hispanic or Latinx': 0,
        'Unknown / Not Reported / Undefined / Declined': -1
    })
    md_select['Race_Grouped'] = md_select['Maternal Race'].replace({
        "Asian,Black or African American": "Multiracial",
        "Asian": "Asian",
        "White": "White",
        "Unknown / Not Reported / Undefined / Declined": "Unknown",
        "Black or African American": "Black",
        "Native Hawaiian or Other Pacific Islander": "Other",
        "American Indian/Alaska Native": "Other"
    })
    md_select = pd.get_dummies(md_select, columns=['Race_Grouped'], prefix='Maternal Race')
    md_select['Cardiac Hypertensive Disease'] = md_select['Cardiac Disease'].astype(str).str.contains('Hypertensive Disease', case=False, na=False).astype(int)
    md_select['ama_custom'] = (md_select['Age at enrollment'] > 35.).astype(int)
    md_select['Diabetes'] = md_select['Diabetes Mellitus'].isin(['Type I','Type II']).astype(int)
    md_select['Past Gestational HTN'] = md_select['Past Pregnancies Diagnoses, Maternal, Hypertension'].isin(['Gestational HTN']).astype(int)
    md_select['Past Gestational diabetes'] = md_select['Past Pregnancies Diagnoses, Maternal'].astype(str).str.contains('Gestational diabetes').astype(int)
    md_select['Past Hypertension'] = md_select['Past Pregnancies Diagnoses, Maternal'].astype(str).str.contains('Hypertension').astype(int)
    md_select = md_select.drop(['Maternal Ethnicity','Maternal Race','Cardiac Disease','Diabetes Mellitus','Past Pregnancies Diagnoses, Maternal, Hypertension','Past Pregnancies Diagnoses, Maternal'],axis=1)

    return(md_select)


def prepare_clinical_file_fmf(md):
    md['Study ID'] = md['Study ID'].str.lower()
    # fill it in if there is a date
    md.loc[md['Date of Hypertension Diagnosis'].notna() & md['Diagnoses, Maternal, Hypertension'].isna(), 'Diagnoses, Maternal, Hypertension'] = 'temp_hypertension'
    md.loc[md['Diagnoses, Maternal, Hypertension, Preeclampsia'].notna() & md['Diagnoses, Maternal, Hypertension'].isna(), 'Diagnoses, Maternal, Hypertension'] = 'temp_hypertension'


    # create df with binary disease columns
    md_df = md[['Study ID','Medical Diagnoses','Cardiac Disease','Has the patient had any eye surgery in the past five years?','Diagnoses, Maternal','Diagnoses, Maternal, Hypertension','Diagnoses, Maternal, Gestational Diabetes',
                'Aspirin?','Statin?','BP meds?','Insulin?','Diagnoses, Maternal, Gestational Diabetes, A2','Tobacco Use']]

    md_df = md_df.rename(columns={'Medical Diagnoses':'med_diag','Cardiac Disease':'card','Has the patient had any eye surgery in the past five years?':'eye_surg','Diagnoses, Maternal':'diag_mat',
                        'Diagnoses, Maternal, Hypertension':'diag_mat_ht', 'Diagnoses, Maternal, Gestational Diabetes':'diag_mat_gd','Diagnoses, Maternal, Gestational Diabetes, A2':'diag_mat_gd_a2'})

    md_df = create_disease_binary_columns(md_df,'med_diag').drop('med_diag', axis=1).drop('med_diag_nonereported',axis=1)
    md_df = create_disease_binary_columns(md_df,'card').drop('card', axis=1)
    md_df = create_disease_binary_columns(md_df,'eye_surg').drop('eye_surg',axis=1).drop('eye_surg_none', axis=1)
    md_df = create_disease_binary_columns(md_df,'diag_mat').drop('diag_mat',axis=1)
    md_df = create_disease_binary_columns(md_df,'diag_mat_ht').drop('diag_mat_ht',axis=1)
    md_df = create_disease_binary_columns(md_df,'diag_mat_gd').drop('diag_mat_gd',axis=1)
    md_df = create_disease_binary_columns(md_df,'Aspirin?').drop('Aspirin?',axis=1).drop('Aspirin?_no',axis=1)
    md_df = create_disease_binary_columns(md_df,'Statin?').drop('Statin?',axis=1).drop('Statin?_no',axis=1)
    md_df = create_disease_binary_columns(md_df,'Insulin?').drop('Insulin?',axis=1).drop('Insulin?_no',axis=1)
    md_df = create_disease_binary_columns(md_df,'BP meds?').drop('BP meds?',axis=1).drop('BP meds?_no',axis=1)
    md_df = create_disease_binary_columns(md_df,'Tobacco Use').drop('Tobacco Use',axis=1)

    md_select = md.merge(md_df,on='Study ID')[['Study ID','Age at enrollment','Tobacco Use_currentsmoker','Past Pregnancies Diagnoses, Maternal, Hypertension, Preeclampsia','diag_mat_ivf','Maternal Ethnicity','Maternal Race',
    'Cardiac Disease','Diabetes Mellitus','Past Pregnancies Diagnoses, Maternal, Hypertension','Past Pregnancies Diagnoses, Maternal','EDD by ultrasound','Term Births', 'Preterm Births','Insulin?']]

    md_select = md_select.rename(columns={'Tobacco Use_currentsmoker':'Current Smoker','Past Pregnancies Diagnoses, Maternal, Hypertension, Preeclampsia':'Past Preeclampsia','diag_mat_ivf':'IVF'})
    md_select['Past Preeclampsia'] = md_select['Past Preeclampsia'].apply(lambda x: 0 if x not in ['without severe features', 'with severe features'] else 1)
    md_select['Maternal hispanic'] = md_select['Maternal Ethnicity'].map({
        'Hispanic or Latinx': 1,
        'NOT Hispanic or Latinx': 0,
        'Unknown / Not Reported / Undefined / Declined': -1
    })
    md_select['Race_Grouped'] = md_select['Maternal Race'].replace({
        "Asian,Black or African American": "Multiracial",
        "Asian": "Asian",
        "White": "White",
        "Unknown / Not Reported / Undefined / Declined": "Unknown",
        "Black or African American": "Black",
        "Native Hawaiian or Other Pacific Islander": "Other",
        "American Indian/Alaska Native": "Other"
    })
    md_select = pd.get_dummies(md_select, columns=['Race_Grouped'], prefix='Maternal Race')
    md_select['Cardiac Hypertensive Disease'] = md_select['Cardiac Disease'].astype(str).str.contains('Hypertensive Disease', case=False, na=False).astype(int)
    md_select['Diabetes_type1'] = md_select['Diabetes Mellitus'].isin(['Type I']).astype(int)
    md_select['Diabetes_type2'] = md_select['Diabetes Mellitus'].isin(['Type II']).astype(int)
    md_select['Insulin?'] = md_select['Insulin?'].apply(lambda x: 1 if x=='Yes' else 0)
    md_select['Past Gestational diabetes'] = md_select['Past Pregnancies Diagnoses, Maternal'].astype(str).str.contains('Gestational diabetes').astype(int)
    md_select['Past Preterm Labor'] = md_select['Past Pregnancies Diagnoses, Maternal'].astype(str).str.contains('Preterm labor').astype(int)
    md_select['nulliparous'] = ((md_select['Term Births'] == 0) & (md_select['Preterm Births'] == 0)).astype(int)
    md_select = md_select.drop(['Maternal Ethnicity','Maternal Race','Cardiac Disease','Diabetes Mellitus','Past Pregnancies Diagnoses, Maternal, Hypertension','Past Pregnancies Diagnoses, Maternal','Term Births', 'Preterm Births'],axis=1)

    return(md_select)


def get_vals(md, col, sets, get_sd=False, round_to=3, compare_to='pw_controls'):
    subj_means = {}
    subj_sds = {}
    subj_vals = {}

    for subj_set in sets.keys():
        subjects = sets[subj_set]
        subset = md[md['Study ID'].str.lower().isin(subjects)][col].dropna()
        mean_val = subset.mean()
        sd_val = subset.std()

        subj_means[subj_set] = mean_val
        subj_sds[subj_set] = sd_val
        subj_vals[subj_set] = subset

        if get_sd:
            print(f"{subj_set}: {mean_val:.{round_to}f} ± {sd_val:.{round_to}f}")
        else:
            print(f"{subj_set}: {mean_val:.{round_to}f}")


    # compare each group against the pw_controls group
    for subj_set in sets.keys():
        if subj_set not in ['cohort', 'pw_controls', 'clean controls']:
            group_vals = subj_vals[subj_set]
            control_vals = subj_vals[compare_to]

            # independent t-test (assumes unequal variances by default)
            t_stat, p_val = mannwhitneyu(group_vals, control_vals, nan_policy='omit')

            print(f"{subj_set} vs {compare_to}: t = {t_stat:.3f}, p = {p_val}")

def print_ast(p_val):
    if p_val > 0.05:
        return('')
    elif p_val > 0.01:
        return('*')
    elif p_val > 0.001:
        return('**')
    else:
        return('***')

def get_vals_count_card(md, col, sets, get_sd=False, compare_to='pw_controls'):
    subj_counts = {}
    subj_total = {}

    for subj_set, subjects in sets.items():
        subset = md[md['Study ID'].str.lower().isin(subjects)]
        subset = subset[subset['Cardiac Disease'].str.lower().str.contains(col, na=False)]
        count_val = len(subset)
        perc = (count_val / len(subjects)) * 100 if len(subjects) > 0 else 0
        print(f"{subj_set}: {count_val} ({perc:.1f}%)")

        subj_counts[subj_set] = count_val
        subj_total[subj_set] = len(subjects)

    for subj_set in sets.keys():
        if subj_set not in ['cohort','pw_controls','clean controls','pw_controls_perf_opt']:

            table = pd.DataFrame({
                "Has Disease": [subj_counts[subj_set], subj_counts[compare_to]],
                "No Disease": [subj_total[subj_set] - subj_counts[subj_set], subj_total[compare_to] - subj_counts[compare_to]]
            }, index=["PEC", "Controls"])
            chi2, p, dof, expected = chi2_contingency(table)
            ast = print_ast(p)
            print(f"{subj_set} vs {compare_to} Chi-square p-value = {p} {ast}")


def get_vals_past_pec(md, col, sets, get_sd=False, compare_to='pw_controls'):
    subj_counts = {}
    subj_total = {}
    for subj_set, subjects in sets.items():
        subset = md[md['Study ID'].str.lower().isin(subjects)]
        subset = subset[subset['Past Pregnancies Diagnoses, Maternal, Hypertension']=='Preeclampsia']
        count_val = len(subset)
        perc = (count_val / len(subjects)) * 100 if len(subjects) > 0 else 0
        print(f"{subj_set}: {count_val} ({perc:.1f}%)")

        subj_counts[subj_set] = count_val
        subj_total[subj_set] = len(subjects)

    for subj_set in sets.keys():
        if subj_set not in ['cohort','pw_controls','clean controls','pw_controls_perf_opt']:

            table = pd.DataFrame({
                "Has Disease": [subj_counts[subj_set], subj_counts[compare_to]],
                "No Disease": [subj_total[subj_set] - subj_counts[subj_set], subj_total[compare_to] - subj_counts[compare_to]]
            }, index=["PEC", "Controls"])
            chi2, p, dof, expected = chi2_contingency(table)
            ast = print_ast(p)
            print(f"{subj_set} vs {compare_to} Chi-square p-value = {p} {ast}")


def get_vals_count_smoker(md, col, sets, get_sd=False, compare_to='pw_controls'):
    subj_counts = {}
    subj_total = {}
    for subj_set, subjects in sets.items():
        subset = md[md['Study ID'].str.lower().isin(subjects)]
        subset = subset[(subset['Tobacco Use']=='Current Smoker') | (subset['Tobacco Use']=='Former Smoker')]
        count_val = len(subset)
        perc = (count_val / len(subjects)) * 100 if len(subjects) > 0 else 0
        print(f"{subj_set}: {count_val} ({perc:.1f}%)")

        subj_counts[subj_set] = count_val
        subj_total[subj_set] = len(subjects)

    for subj_set in sets.keys():
        if subj_set not in ['cohort','pw_controls','clean controls','pw_controls_perf_opt']:

            table = pd.DataFrame({
                "Has Disease": [subj_counts[subj_set], subj_counts[compare_to]],
                "No Disease": [subj_total[subj_set] - subj_counts[subj_set], subj_total[compare_to] - subj_counts[compare_to]]
            }, index=["PEC", "Controls"])
            chi2, p, dof, expected = chi2_contingency(table)
            ast = print_ast(p)
            print(f"{subj_set} vs {compare_to} Chi-square p-value = {p} {ast}")



def get_vals_count_ethnicity(md, col, sets, get_sd=False, compare_to='pw_controls'):
    subj_counts = {}
    subj_total = {}
    for subj_set, subjects in sets.items():
        subset = md[md['Study ID'].str.lower().isin(subjects)]
        subset = subset[subset['Maternal Ethnicity']=='Hispanic or Latinx']
        count_val = len(subset)
        perc = (count_val / len(subjects)) * 100 if len(subjects) > 0 else 0
        print(f"{subj_set}: {count_val} ({perc:.1f}%)")

        subj_counts[subj_set] = count_val
        subj_total[subj_set] = len(subjects)

    for subj_set in sets.keys():
        if subj_set not in ['cohort','pw_controls','clean controls','pw_controls_perf_opt']:

            table = pd.DataFrame({
                "Has Disease": [subj_counts[subj_set], subj_counts[compare_to]],
                "No Disease": [subj_total[subj_set] - subj_counts[subj_set], subj_total[compare_to] - subj_counts[compare_to]]
            }, index=["PEC", "Controls"])
            chi2, p, dof, expected = chi2_contingency(table)
            ast = print_ast(p)
            print(f"{subj_set} vs {compare_to} Chi-square p-value = {p} {ast}")


def get_vals_count_db(md, col, sets, get_sd=False, compare_to='pw_controls'):
    subj_counts = {}
    subj_total = {}
    for subj_set, subjects in sets.items():
        subset = md[md['Study ID'].str.lower().isin(subjects)]
        subset = subset[subset['Diabetes Mellitus'].isna()==False]
        count_val = len(subset)
        perc = (count_val / len(subjects)) * 100 if len(subjects) > 0 else 0
        print(f"{subj_set}: {count_val} ({perc:.1f}%)")

        subj_counts[subj_set] = count_val
        subj_total[subj_set] = len(subjects)

    for subj_set in sets.keys():
        if subj_set not in ['cohort','pw_controls','clean controls','pw_controls_perf_opt']:

            table = pd.DataFrame({
                "Has Disease": [subj_counts[subj_set], subj_counts[compare_to]],
                "No Disease": [subj_total[subj_set] - subj_counts[subj_set], subj_total[compare_to] - subj_counts[compare_to]]
            }, index=["PEC", "Controls"])
            chi2, p, dof, expected = chi2_contingency(table)
            ast = print_ast(p)
            print(f"{subj_set} vs {compare_to} Chi-square p-value = {p} {ast}")


def get_vals_count_diagmat(md, col, sets, get_sd=False, compare_to='pw_controls'):
    subj_counts = {}
    subj_total = {}
    for subj_set, subjects in sets.items():
        subset = md[md['Study ID'].str.lower().isin(subjects)]
        subset = subset[subset['Diagnoses, Maternal'].str.lower().str.contains(col, na=False)]
        count_val = len(subset)
        perc = (count_val / len(subjects)) * 100 if len(subjects) > 0 else 0
        print(f"{subj_set}: {count_val} ({perc:.1f}%)")

        subj_counts[subj_set] = count_val
        subj_total[subj_set] = len(subjects)

    for subj_set in sets.keys():
        if subj_set not in ['cohort','pw_controls','clean controls','pw_controls_perf_opt']:

            table = pd.DataFrame({
                "Has Disease": [subj_counts[subj_set], subj_counts[compare_to]],
                "No Disease": [subj_total[subj_set] - subj_counts[subj_set], subj_total[compare_to] - subj_counts[compare_to]]
            }, index=["PEC", "Controls"])
            chi2, p, dof, expected = chi2_contingency(table)
            ast = print_ast(p)
            print(f"{subj_set} vs {compare_to} Chi-square p-value = {p} {ast}")


def get_vals_count_meddiag(md, col, sets, get_sd=False, compare_to='pw_controls'):
    subj_counts = {}
    subj_total = {}
    for subj_set, subjects in sets.items():
        subset = md[md['Study ID'].str.lower().isin(subjects)]
        subset = subset[subset['Medical Diagnoses'].str.lower().str.contains(col, na=False)]
        count_val = len(subset)
        perc = (count_val / len(subjects)) * 100 if len(subjects) > 0 else 0
        print(f"{subj_set}: {count_val} ({perc:.1f}%)")

        subj_counts[subj_set] = count_val
        subj_total[subj_set] = len(subjects)

    for subj_set in sets.keys():
        if subj_set not in ['cohort','pw_controls','clean controls','pw_controls_perf_opt']:

            table = pd.DataFrame({
                "Has Disease": [subj_counts[subj_set], subj_counts[compare_to]],
                "No Disease": [subj_total[subj_set] - subj_counts[subj_set], subj_total[compare_to] - subj_counts[compare_to]]
            }, index=["PEC", "Controls"])
            chi2, p, dof, expected = chi2_contingency(table)
            ast = print_ast(p)
            print(f"{subj_set} vs {compare_to} Chi-square p-value = {p} {ast}")


def get_vals_count_np(md, col, sets, get_sd=False, compare_to='pw_controls'):
    subj_counts = {}
    subj_total = {}
    for subj_set, subjects in sets.items():
        subset = md[md['Study ID'].str.lower().isin(subjects)]
        subset = subset[subset['Living Children']==0]
        count_val = len(subset)
        perc = (count_val / len(subjects)) * 100 if len(subjects) > 0 else 0
        print(f"{subj_set}: {count_val} ({perc:.1f}%)")

        subj_counts[subj_set] = count_val
        subj_total[subj_set] = len(subjects)

    for subj_set in sets.keys():
        if subj_set not in ['cohort','pw_controls','clean controls','pw_controls_perf_opt']:

            table = pd.DataFrame({
                "Has Disease": [subj_counts[subj_set], subj_counts[compare_to]],
                "No Disease": [subj_total[subj_set] - subj_counts[subj_set], subj_total[compare_to] - subj_counts[compare_to]]
            }, index=["PEC", "Controls"])
            chi2, p, dof, expected = chi2_contingency(table)
            ast = print_ast(p)
            print(f"{subj_set} vs {compare_to} Chi-square p-value = {p} {ast}")


def get_bmi(df: pd.DataFrame, weight_col: str, height_col: str) -> pd.DataFrame:
    df['bmi'] = df[weight_col]/(df[height_col]**2)
    return(df)


def proc_birth_hx(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            if 'healthy' in x.lower():
                return(0)
            elif 'pe' in x.lower():
                return(1)     
            elif 'gdm' in x.lower():
                return(2)
            elif 'ghtn' in x.lower():
                return(3)
               
    df[col] = df[col].apply(parse)
    df['Past Preeclampsia'] = df[col].apply(lambda x: 1 if x==1 else 0)
    df['Past Gestational HTN'] = df[col].apply(lambda x: 1 if x==3 else 0)
    df['Past Gestational diabetes'] = df[col].apply(lambda x: 1 if x==2 else 0)
    
    return df


def proc_on_yes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            if 'yes' in x.lower():
                return(1)
            else:
                return(0)
        
    df[col] = df[col].apply(parse)
    return df  


def proc_on_no(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            if 'no' in x.lower():
                return(0)
            else:
                return(1)
        
    df[col] = df[col].apply(parse)
    return df


def proc_smoker(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].apply(lambda x: 1 if 'former' in x.lower() else 0)
    return df


def proc_db(df: pd.DataFrame, col: str) -> pd.DataFrame:
    
    def parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            if '1' in x:
                return 1
            elif '2' in x:
                return 1 # NOT INTERESTED IN TYPES
            else:
                return 0
        
    df[col] = df[col].apply(parse)
    return(df)


def proc_conception(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df['IVF'] = df[col].apply(lambda x: 1 if 'ivf' in x.lower() and 'natural' not in x.lower() else 0)
    return df    


def compare_binary_col(
    nyu_md,
    col,
    controls_ids,
    group_ids,
    id_col="Record ID"
):
    """
    nyu_md : DataFrame
    col : str
        Binary column to compare.
    controls_ids : list
        IDs for controls.
    group_ids : dict
        Dict of group_name -> list_of_ids.
    """
    ids = nyu_md[id_col].str.lower()
    controls = nyu_md[ids.isin(controls_ids)]
    control_yes = int(controls[col].sum())
    control_no = len(controls) - control_yes
    print(f"\n=== {col} ===")
    print(f"controls: {control_yes}/{len(controls)} "
          f"({control_yes/len(controls):.3f})")
    results = []
    for group_name, group_list in group_ids.items():
        df = nyu_md[ids.isin(group_list)]
        yes = int(df[col].sum())
        no = len(df) - yes
        prop = yes / len(df)

        odds_ratio, pval = fisher_exact([
            [control_yes, control_no],
            [yes, no]
        ])

        print(f"{group_name}: {yes}/{len(df)} ({prop:.3f}) pval: {pval}")
        results.append({
            "group": group_name,
            "n": len(df),
            "count": yes,
            "proportion": prop,
            "odds_ratio": odds_ratio,
            "pvalue": pval
        })
    return pd.DataFrame(results)


def compare_continuous_col(
    nyu_md,
    col,
    controls_ids,
    group_ids,
    id_col="Record ID",
    test="mannwhitney"  # "ttest" or "mannwhitney"
):
    """
    Compare a continuous column between control and each group.

    nyu_md : DataFrame
    col : str
        Continuous column to compare.
    controls_ids : list
        IDs for controls.
    group_ids : dict
        Dict of group_name -> list_of_ids.
    id_col : str
    test : str
        "ttest" (Welch's t-test) or "mannwhitney".
    """
    ids = nyu_md[id_col].str.lower()

    control_vals = nyu_md.loc[ids.isin(controls_ids), col].dropna()

    print(f"\n=== {col} ===")
    print(f"controls: n={len(control_vals)}, "
          f"mean={control_vals.mean():.3f}, "
          f"median={control_vals.median():.3f}, "
          f"sd={control_vals.std():.3f}")

    results = []
    for group_name, group_list in group_ids.items():
        group_vals = nyu_md.loc[ids.isin(group_list), col].dropna()

        if test == "ttest":
            stat, pval = ttest_ind(
                control_vals, group_vals, equal_var=False  # Welch's
            )
        elif test == "mannwhitney":
            stat, pval = mannwhitneyu(
                control_vals, group_vals, alternative="two-sided"
            )
        else:
            raise ValueError("test must be 'ttest' or 'mannwhitney'")

        print(f"{group_name}: n={len(group_vals)}, "
              f"mean={group_vals.mean():.3f}, "
              f"median={group_vals.median():.3f}, "
              f"sd={group_vals.std():.3f}, "
              f"pval: {pval}")

        results.append({
            "group": group_name,
            "n": len(group_vals),
            "mean": group_vals.mean(),
            "median": group_vals.median(),
            "sd": group_vals.std(),
            "statistic": stat,
            "pvalue": pval,
            "test": test,
        })

    return pd.DataFrame(results)


def classification_metrics(cases_pos, cases_neg, controls_pos, controls_neg):
    """
    Compute classification metrics from case/control predicted counts.
 
    Parameters
    cases_pos    : int | float  — cases predicted positive    (True Positives)
    cases_neg    : int | float  — cases predicted negative    (False Negatives)
    controls_pos : int | float  — controls predicted positive (False Positives)
    controls_neg : int | float  — controls predicted negative (True Negatives)
 
    Returns
    dict with keys:
        TP, FP, TN, FN,
        TPR  (sensitivity / recall),
        FPR  (fall-out),
        PPV  (precision),
        NPV,
        F1
    """
    TP = cases_pos
    FN = cases_neg
    FP = controls_pos
    TN = controls_neg
 
    TPR = TP / (TP + FN) if (TP + FN) > 0 else float("nan")   # sensitivity
    FPR = FP / (FP + TN) if (FP + TN) > 0 else float("nan")   # fall-out
    PPV = TP / (TP + FP) if (TP + FP) > 0 else float("nan")   # precision
    NPV = TN / (TN + FN) if (TN + FN) > 0 else float("nan")
    F1  = (2 * PPV * TPR) / (PPV + TPR) if (PPV + TPR) > 0 else float("nan")
 
    return {
        "TP":  TP,
        "FP":  FP,
        "TN":  TN,
        "FN":  FN,
        "TPR": TPR,   # sensitivity / recall
        "FPR": FPR,
        "PPV": PPV,   # precision
        "NPV": NPV,
        "F1":  F1,
    }

        
def strings_to_na(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def gest_age_to_days(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert gestational age strings like '12w1d', '9w', '3d'
    into total days as a numeric column.
    Invalid or missing values -> NaN.
    """
    pattern = re.compile(r'^\s*(?:(\d+)\s*w)?\s*(?:(\d+)\s*d)?\s*$', re.I)

    def parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return x  # already numeric (assume days)
        
        if ' ' in x:
            x = x.split(' ')[0]
        m = pattern.match(str(x))
        if not m:
            return np.nan
        weeks = int(m.group(1)) if m.group(1) else 0
        days = int(m.group(2)) if m.group(2) else 0
        return weeks * 7 + days
    
    df = df.copy()

    df[col] = df[col].apply(parse).astype('float')
    return df


def datetime_to_day_index(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert a datetime column into integer days since the earliest date.
    Earliest date -> 0, next day -> 1, etc.
    """
    # Ensure datetime
    dt = pd.to_datetime(df[col], errors='coerce')

    # Normalize to date (drop time-of-day)
    dt = dt.dt.normalize()

    # Compute day index
    df[col] = (dt - dt.min()).dt.days

    return df


def process_ethnicity(eth: str):
    """
    Parse a free-text ethnicity string into:
    - race: White / Black / Asian / Multiracial / Other / Unknown
    - ethnicity: hispanic / not hispanic
    """
    if not isinstance(eth, str) or not eth.strip():
        return "Unknown", "not hispanic"

    s = eth.lower()
    
    if eth == 'laotian':
        return('asian', 'not hispanic')

    races = ['white', 'black', 'asian']
    found = [r for r in races if r in s]

    # Determine race
    if len(found) > 1:
        race = 'Multiracial'
    elif len(found) == 1:
        race = found[0].capitalize()
    else:
        race = 'Other'

    # Determine ethnicity
    if 'hispanic' in s and 'not' not in s:
        ethnicity = 'hispanic'
    else:
        ethnicity = 'not hispanic'

    return race, ethnicity


def add_ethnicity_binaries(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    From a free-text ethnicity column, add binary columns:
    White, Black, Asian, Multiracial, Hispanic
    """
    parsed = df[col].apply(process_ethnicity)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=['race', 'ethnicity'], index=df.index)

    race = parsed_df['race'].fillna('').str.lower()
    ethnicity = parsed_df['ethnicity'].fillna('').str.lower()

    df['Maternal Race_White'] = (race == 'white').astype(int)
    df['Maternal Race_Black'] = (race == 'black').astype(int)
    df['Maternal Race_Asian'] = (race == 'asian').astype(int)
    df['Maternal Race_Multiracial'] = (race == 'multiracial').astype(int)
    df['Maternal Race_Other'] = (~race.isin(['white', 'black', 'asian', 'multiracial'])).astype(int)

    df['Maternal hispanic'] = (ethnicity == 'hispanic').astype(int)

    return df


def get_ama(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df['ama'] = (df[col] > 35.).astype(int)
    df['ama_custom'] = (df[col] > 35.).astype(int)
    return(df)
