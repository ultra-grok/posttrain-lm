import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np

# --- Configuration ---
HUB_DATASET_ID = "ultra-grok/tldr_sft_genreverse"  # << changed
ADAPTER_REVISION = "2sft"
OUTPUT_FILENAME = "len_"+ADAPTER_REVISION+"_distribution_reverse.png"  # << optional

def plot_score_distribution():
    """
    Loads the dataset, plots the distribution of the 'score' column,
    and saves the plot to a file.
    """
    try:
        # Load the dataset from the Hugging Face Hub
        print(f"Loading dataset '{HUB_DATASET_ID}' (revision: {ADAPTER_REVISION})...")
        dataset = load_dataset(HUB_DATASET_ID, split="validation", revision=ADAPTER_REVISION)

        # Extract the scores into a list
        scores = [len(example['completion']) for example in dataset]

        # --- Plotting ---
        print("Generating plot...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the histogram
        ax.hist(scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

        # Add titles and labels for clarity
        ax.set_title('Distribution of Completion Lengths', fontsize=16, fontweight='bold')
        ax.set_xlabel('Length of Completion', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        # Add a vertical line at the mean
        mean_score = np.mean(scores)
        var_score = np.var(scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
        
        # Add text box for mean and variance
        textstr = '\n'.join((
            r'$\mu=%.2f$' % (mean_score, ),
            r'$\sigma^2=%.2f$' % (var_score, )))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        ax.legend()
        
        # Improve layout and save the figure
        plt.tight_layout()
        plt.savefig(OUTPUT_FILENAME)

        print(f"Plot successfully saved as '{OUTPUT_FILENAME}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_score_distribution()
