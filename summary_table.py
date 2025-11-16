#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a summary table of statistical test results
"""

import pandas as pd
import numpy as np

def create_summary_table():
    """Create a summary table of key findings."""
    
    # Read the statistical test results
    df = pd.read_csv('statistical_test_results.csv')
    
    # Focus on key metrics
    key_metrics = ['AUC-PR', 'AUC-ROC', 'Standard-F1', 'VUS-PR']
    
    summary_data = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        for metric in key_metrics:
            metric_df = dataset_df[dataset_df['metric'] == metric]
            if len(metric_df) == 0:
                continue
            
            row = metric_df.iloc[0]
            
            # Determine winner
            if row['mean_difference'] > 0:
                winner = 'Time_RCD'
                diff = row['mean_difference']
            else:
                winner = 'DADA'
                diff = abs(row['mean_difference'])
            
            # Get significance
            p_value = row['paired_t_pvalue'] if pd.notna(row['paired_t_pvalue']) else row['independent_t_pvalue']
            
            if pd.isna(p_value):
                significance = 'N/A'
            elif p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'
            
            summary_data.append({
                'Dataset': dataset,
                'Metric': metric,
                'Winner': winner,
                'DADA Mean': f"{row['dada_mean']:.3f}",
                'Time_RCD Mean': f"{row['timercd_mean']:.3f}",
                'Difference': f"{diff:.3f}",
                'Significance': significance,
                'P-value': f"{p_value:.4f}" if pd.notna(p_value) else 'N/A',
                'Sample Size': int(row['dada_n'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv('summary_table.csv', index=False)
    
    # Print summary by dataset
    print("="*100)
    print("SUMMARY BY DATASET")
    print("="*100)
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_summary = summary_df[summary_df['Dataset'] == dataset]
        if len(dataset_summary) == 0:
            continue
        
        print(f"\n{dataset} (n={dataset_summary.iloc[0]['Sample Size']}):")
        print("-" * 100)
        
        time_rcd_wins = len(dataset_summary[dataset_summary['Winner'] == 'Time_RCD'])
        dada_wins = len(dataset_summary[dataset_summary['Winner'] == 'DADA'])
        
        print(f"  Time_RCD wins: {time_rcd_wins}/{len(dataset_summary)} metrics")
        print(f"  DADA wins: {dada_wins}/{len(dataset_summary)} metrics")
        
        for _, row in dataset_summary.iterrows():
            print(f"    {row['Metric']:20s}: {row['Winner']:10s} (diff={row['Difference']:6s}, p={row['P-value']:8s} {row['Significance']})")
    
    # Overall summary
    print("\n" + "="*100)
    print("OVERALL SUMMARY")
    print("="*100)
    
    overall_time_rcd = len(summary_df[summary_df['Winner'] == 'Time_RCD'])
    overall_dada = len(summary_df[summary_df['Winner'] == 'DADA'])
    
    print(f"\nTime_RCD wins: {overall_time_rcd}/{len(summary_df)} comparisons")
    print(f"DADA wins: {overall_dada}/{len(summary_df)} comparisons")
    
    # By dataset winner
    print("\n" + "="*100)
    print("DATASET-LEVEL WINNERS (based on AUC-PR)")
    print("="*100)
    
    auc_pr_summary = summary_df[summary_df['Metric'] == 'AUC-PR']
    time_rcd_datasets = auc_pr_summary[auc_pr_summary['Winner'] == 'Time_RCD']['Dataset'].tolist()
    dada_datasets = auc_pr_summary[auc_pr_summary['Winner'] == 'DADA']['Dataset'].tolist()
    
    print(f"\nTime_RCD wins on: {', '.join(sorted(time_rcd_datasets))}")
    print(f"DADA wins on: {', '.join(sorted(dada_datasets))}")
    
    print(f"\nSummary table saved to: summary_table.csv")
    
    return summary_df

if __name__ == "__main__":
    summary_df = create_summary_table()

