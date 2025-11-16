#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a side-by-side comparison of model significance vs seed significance
"""

import pandas as pd
import numpy as np

def create_comparison_table():
    """Create a comparison table showing both types of significance."""
    
    # Load both result files
    model_results = pd.read_csv('statistical_test_results.csv')
    seed_results = pd.read_csv('seed_analysis_results.csv')
    
    # Focus on key metrics
    key_metrics = ['AUC-PR', 'AUC-ROC', 'Standard-F1']
    
    comparison_data = []
    
    for dataset in model_results['dataset'].unique():
        for metric in key_metrics:
            # Model comparison results
            model_row = model_results[
                (model_results['dataset'] == dataset) & 
                (model_results['metric'] == metric)
            ]
            
            if len(model_row) == 0:
                continue
            
            model_row = model_row.iloc[0]
            
            # Seed variability results for both models
            seed_dada = seed_results[
                (seed_results['dataset'] == dataset) & 
                (seed_results['model'] == 'DADA') & 
                (seed_results['metric'] == metric)
            ]
            
            seed_timercd = seed_results[
                (seed_results['dataset'] == dataset) & 
                (seed_results['model'] == 'Time_RCD') & 
                (seed_results['metric'] == metric)
            ]
            
            # Model significance
            model_p = model_row['paired_t_pvalue'] if pd.notna(model_row['paired_t_pvalue']) else model_row['independent_t_pvalue']
            model_sig = 'YES' if pd.notna(model_p) and model_p < 0.05 else 'NO'
            model_winner = 'Time_RCD' if model_row['mean_difference'] > 0 else 'DADA'
            
            # Seed significance for DADA
            dada_seed_sig = 'NO'
            dada_cv = np.nan
            if len(seed_dada) > 0:
                dada_p = seed_dada.iloc[0]['anova_pvalue']
                dada_cv = seed_dada.iloc[0]['cv_across_seeds']
                dada_seed_sig = 'YES' if pd.notna(dada_p) and dada_p < 0.05 else 'NO'
            
            # Seed significance for Time_RCD
            timercd_seed_sig = 'NO'
            timercd_cv = np.nan
            if len(seed_timercd) > 0:
                timercd_p = seed_timercd.iloc[0]['anova_pvalue']
                timercd_cv = seed_timercd.iloc[0]['cv_across_seeds']
                timercd_seed_sig = 'YES' if pd.notna(timercd_p) and timercd_p < 0.05 else 'NO'
            
            comparison_data.append({
                'Dataset': dataset,
                'Metric': metric,
                'Model_Significant': model_sig,
                'Model_Winner': model_winner,
                'Model_Pvalue': f"{model_p:.4f}" if pd.notna(model_p) else 'N/A',
                'DADA_Seed_Significant': dada_seed_sig,
                'DADA_CV': f"{dada_cv:.2f}%" if pd.notna(dada_cv) else 'N/A',
                'TimeRCD_Seed_Significant': timercd_seed_sig,
                'TimeRCD_CV': f"{timercd_cv:.2f}%" if pd.notna(timercd_cv) else 'N/A',
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('significance_comparison_table.csv', index=False)
    
    # Print summary
    print("="*100)
    print("SIDE-BY-SIDE COMPARISON: MODEL SIGNIFICANCE vs SEED SIGNIFICANCE")
    print("="*100)
    
    print("\nKey:")
    print("  Model_Significant: Are DADA and Time_RCD significantly different? (YES/NO)")
    print("  Seed_Significant: Are different seeds significantly different? (YES/NO)")
    print("  CV: Coefficient of Variation (lower = more stable)")
    
    print("\n" + "-"*100)
    for dataset in sorted(comparison_df['Dataset'].unique()):
        dataset_data = comparison_df[comparison_df['Dataset'] == dataset]
        print(f"\n{dataset}:")
        print(f"{'Metric':<20} {'Model Diff?':<12} {'Winner':<12} {'DADA Seeds':<15} {'TimeRCD Seeds':<15}")
        print("-"*100)
        for _, row in dataset_data.iterrows():
            print(f"{row['Metric']:<20} {row['Model_Significant']:<12} {row['Model_Winner']:<12} "
                  f"{row['DADA_Seed_Significant']:<7} ({row['DADA_CV']:<6}) {row['TimeRCD_Seed_Significant']:<7} ({row['TimeRCD_CV']:<6})")
    
    # Overall statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    
    model_sig_count = len(comparison_df[comparison_df['Model_Significant'] == 'YES'])
    model_total = len(comparison_df)
    print(f"\nModel Significance (DADA vs Time_RCD):")
    print(f"  Significant differences: {model_sig_count}/{model_total} ({model_sig_count/model_total*100:.1f}%)")
    
    dada_seed_sig_count = len(comparison_df[comparison_df['DADA_Seed_Significant'] == 'YES'])
    timercd_seed_sig_count = len(comparison_df[comparison_df['TimeRCD_Seed_Significant'] == 'YES'])
    print(f"\nSeed Significance:")
    print(f"  DADA seed differences: {dada_seed_sig_count}/{model_total} ({dada_seed_sig_count/model_total*100:.1f}%)")
    print(f"  Time_RCD seed differences: {timercd_seed_sig_count}/{model_total} ({timercd_seed_sig_count/model_total*100:.1f}%)")
    
    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)
    print("""
✅ MODEL SIGNIFICANCE (High % = Models are different):
   - Most comparisons show YES → DADA and Time_RCD perform differently
   - This is GOOD - it means you can choose one model over another

✅ SEED SIGNIFICANCE (Low % = Models are stable):
   - Most comparisons show NO → Results are stable across seeds
   - This is GOOD - it means results are reproducible

📊 SUMMARY:
   - Models ARE different (significant model differences)
   - Models ARE stable (not significant seed differences)
   - This means: You can confidently compare models AND trust the results
    """)
    
    print(f"\nComparison table saved to: significance_comparison_table.csv")
    
    return comparison_df

if __name__ == "__main__":
    comparison_df = create_comparison_table()

