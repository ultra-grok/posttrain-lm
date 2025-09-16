import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
"""
PROJECT OVERVIEW: Self-Judging AI for Text Generation
------------------------------------------------------
This script visualizes the results of a project aimed at creating a self-improving text
generation system. The core idea is that judging text is an easier task than generating it.
Therefore, we can use a "critic" model to score synthetically generated outputs and filter for
the highest quality examples.

This script's main purpose is to generate the "defining plot" of the project, which evaluates
how well the critic's score (proxied by the sum of the last 5 token logits) correlates with a
final quality score from a powerful, external model (Google's Gemini 2.5 Pro).

DATASET CREATION & PROVENANCE:
------------------------------
The ratings in this dataset were generated using the Gemini 2.5 Pro API. Due to a free-tier
daily quota limit of 50 requests, the rating process was done in two main batches:
1.  Batch 1: Rated posts starting from the first row forward.
2.  Batch 2: Rated posts starting from the last row backward to maximize coverage.

The qualitative 'gemini_reasoning' from the first forward pass was unfortunately lost, the resulting
dataset was pushed to the Hugging Face Hub under the revision 'merged_conditional_ratings_v1'.

SUMMARY OF KEY FINDINGS (from the final statistics table):
-----------------------------------------------------------
The analysis reveals a strong but nuanced relationship between the critic's score and the
final quality rating, which is heavily dependent on the length of the generated text.

* Q1 (Shortest Outputs): Shows the most dramatic effect. The mean rating jumps from a
    baseline of 54.5 to 89.0 for the top 5% of critic-scored posts, a ~35-point increase.
    - Baseline (n=155): Mean Rating 54.5
    - Top 20% (n=31):  Mean Rating 67.9
    - Top 10% (n=16):  Mean Rating 72.7
    - Top 5% (n=8):   Mean Rating 89.0

* Q2 (Med-Short Outputs): Displays a strong positive trend. The mean rating rises from
    a baseline of 57.2 to 79.1 for the top 5%.
    - Baseline (n=155): Mean Rating 57.2
    - Top 20% (n=31):  Mean Rating 64.5
    - Top 10% (n=16):  Mean Rating 71.5
    - Top 5% (n=8):   Mean Rating 79.1

* Q3 (Med-Long Outputs): Also shows a significant positive correlation. The mean
    rating increases from a baseline of 55.0 to 80.8 for the top 5%.
    - Baseline (n=154): Mean Rating 55.0
    - Top 20% (n=31):  Mean Rating 75.4
    - Top 10% (n=16):  Mean Rating 79.6
    - Top 5% (n=8):   Mean Rating 80.8

* Q4 (Longest Outputs): Reveals the breakdown in the pattern. The critic's score
    becomes a neutral or negligible indicator. The mean rating for the top 5% (48.5)
    is slightly lower than the baseline (50.2).
    - Baseline (n=155): Mean Rating 50.2
    - Top 20% (n=31):  Mean Rating 55.2
    - Top 10% (n=16):  Mean Rating 49.6
    - Top 5% (n=8):   Mean Rating 48.5
"""
# --- 1. Configuration ---
# All settings are at the top for easy modification.
REPO_ID = "ultra-grok/tldr_sft_genreverse"
MERGED_REVISION = "merged_conditional_ratings_v1"
OUTPUT_FILENAME = "self_judging_ai_final_plot.png"

def main():
    """
    Main function to run the complete analysis and visualization pipeline.
    """
    # --- 2. Load and Prepare the Dataset ---
    print(f"--- Step 1: Loading and preparing data from Hub: '{REPO_ID}' ---")
    
    # Load the specified revision of the dataset
    dataset = load_dataset(REPO_ID, revision=MERGED_REVISION, split="train")
    df = dataset.to_pandas()

    # Calculate 'last_label' for filtering
    df['last_label'] = df['labels'].apply(lambda x: x[-1])
    
    # Filter for valid ratings and last_label == 0
    df_filtered = df[(df['gemini_rating'] != -1) & (df['last_label'] == 0)].copy()
    print(f"Filtered data down to {len(df_filtered)} rows.")
    
    # Calculate features needed for analysis
    df_filtered['completion_length'] = df_filtered['completion'].str.len()
    df_filtered['sum_of_last_5_logits'] = df_filtered['token_logits'].apply(lambda x: sum(x[-5:]))
    
    # Stratify data into 4 length quartiles
    df_filtered['length_quartile'] = pd.qcut(
        df_filtered['completion_length'],
        q=4,
        labels=['Q1 (Shortest Outputs)', 'Q2 (Med-Short Outputs)', 'Q3 (Med-Long Outputs)', 'Q4 (Longest Outputs)']
    )
    
    # --- 3. Generate the Summary Statistics (Dynamically) ---
    print("--- Step 2: Calculating summary statistics for each stratum ---")
    results = []
    logit_percentiles = {'Top 5%': 0.95, 'Top 10%': 0.90, 'Top 20%': 0.80}

    # Group by each length quartile to analyze separately
    for name, group in df_filtered.groupby('length_quartile', observed=True):
        # Calculate the baseline stats for the entire quartile
        baseline_stats = group['gemini_rating'].describe()
        results.append({
            'Length Quartile': name, 'Logit Percentile': 'Baseline (All)',
            'Mean Rating': baseline_stats['mean'], 'Std Dev Rating': baseline_stats['std'],
            'Sample Size': int(baseline_stats['count'])
        })
        
        # Calculate stats for the top logit percentiles
        for percentile_label, quantile_value in logit_percentiles.items():
            threshold = group['sum_of_last_5_logits'].quantile(quantile_value)
            subset_df = group[group['sum_of_last_5_logits'] >= threshold]
            stats = subset_df['gemini_rating'].describe()
            results.append({
                'Length Quartile': name, 'Logit Percentile': percentile_label,
                'Mean Rating': stats['mean'], 'Std Dev Rating': stats['std'],
                'Sample Size': int(stats['count'])
            })

    df_plot = pd.DataFrame(results)

    # --- 4. Generate the Final Plot ---
    print("--- Step 3: Generating the final plot ---")
    
    # Calculate the Standard Error of the Mean (SEM) for the error bars
    df_plot['sem'] = df_plot['Std Dev Rating'] / np.sqrt(df_plot['Sample Size'])

    # Set a logical order for the x-axis categories
    logit_order = ['Baseline (All)', 'Top 20%', 'Top 10%', 'Top 5%']
    df_plot['Logit Percentile'] = pd.Categorical(df_plot['Logit Percentile'], categories=logit_order, ordered=True)

    # Create the 2x2 plot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=True)
    fig.suptitle("Evaluating a Self-Judging AI: Critic's Score vs. Final Quality", fontsize=20, y=1.03)
    axes = axes.flatten()

    quartile_order = ['Q1 (Shortest Outputs)', 'Q2 (Med-Short Outputs)', 'Q3 (Med-Long Outputs)', 'Q4 (Longest Outputs)']

    for i, quartile in enumerate(quartile_order):
        ax = axes[i]
        quartile_data = df_plot[df_plot['Length Quartile'] == quartile]
        
        bars = sns.barplot(
            data=quartile_data, x='Logit Percentile', y='Mean Rating', ax=ax,
            hue='Logit Percentile', palette='viridis', legend=False, order=logit_order
        )
        
        quartile_data_sorted = quartile_data.set_index('Logit Percentile').reindex(logit_order)
        
        ax.errorbar(
            x=np.arange(len(logit_order)), y=quartile_data_sorted['Mean Rating'],
            yerr=quartile_data_sorted['sem'], fmt='none', c='black', capsize=5
        )

        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points',
                fontweight='bold', fontsize=11
            )
        
        total_n = quartile_data[quartile_data['Logit Percentile'] == 'Baseline (All)']['Sample Size'].iloc[0]
        ax.text(0.95, 0.95, f'n = {total_n}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', lw=1, alpha=0.8))

        ax.set_title(quartile, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_ylim(0, 105)
        ax.tick_params(axis='x', rotation=25, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

    fig.text(0.5, -0.02, "Critic's Score Percentile Group", ha='center', fontsize=14)
    fig.text(-0.02, 0.5, "Mean Quality Score (Gemini 2.5 Pro)", va='center', rotation='vertical', fontsize=14)
    footnote_text = "Task: Evaluating synthetically generated TL;DR summaries of Reddit posts."
    fig.text(0.99, 0.01, footnote_text, ha='right', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Plot successfully saved to '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    main()
