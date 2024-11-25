import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main1(prior_fp, posterior_fp, output_dir):
    df1 = pd.read_csv(prior_fp)
    df2 = pd.read_csv(posterior_fp)

    for model_name in df1["model_name"].unique():
        model_df1 = df1[df1["model_name"] == model_name]
        model_df2 = df2[df2["model_name"] == model_name]

        prior_a = model_df1["p(A)"].dropna().tolist()
        prior_b = model_df1["p(B)"].dropna().tolist()
        prior_c = model_df1["p(C)"].dropna().tolist()

        posterior_a = model_df2["p(A|E)"].dropna().tolist()
        posterior_b = model_df2["p(B|E)"].dropna().tolist()
        posterior_c = model_df2["p(C|E)"].dropna().tolist()

        prior_means = [np.mean(prior_a)/100, np.mean(prior_b)/100, np.mean(prior_c)/100]
        posterior_means = [np.mean(posterior_a), np.mean(posterior_b), np.mean(posterior_c)]
        categories = ['Agricultural and Veterinary Science', 'Business Administration', 'Computer Science']

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(prior_means, posterior_means, s=100, edgecolor='black', alpha=0.7)

        # Annotate points with categories
        for i, category in enumerate(categories):
            plt.annotate(category, (prior_means[i], posterior_means[i]-0.01), fontsize=10, ha='center', va='top')

        # Customize plot appearance
        plt.xlabel('Mean Prior Probability', fontsize=12)
        plt.ylabel('Mean Posterior Probability', fontsize=12)
        plt.title(f'{model_name}', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
        plt.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
        plt.tight_layout()

        output_fp = os.path.join(output_dir, f'{model_name}.png')
        plt.savefig(output_fp, bbox_inches='tight', dpi=300)
        plt.close()


def main2(similarity_fp, posterior_fp, output_dir):
    df1 = pd.read_csv(similarity_fp)
    df2 = pd.read_csv(posterior_fp)

    for model_name in df1["model_name"].unique():
        model_df1 = df1[df1["model_name"] == model_name]
        model_df2 = df2[df2["model_name"] == model_name]

        sim_a = model_df1["sim_A"].dropna().tolist()
        sim_b = model_df1["sim_B"].dropna().tolist()
        sim_c = model_df1["sim_C"].dropna().tolist()

        posterior_a = model_df2["p(A|E)"].dropna().tolist()
        posterior_b = model_df2["p(B|E)"].dropna().tolist()
        posterior_c = model_df2["p(C|E)"].dropna().tolist()

        sim_means = [np.mean(sim_a), np.mean(sim_b), np.mean(sim_c)]
        posterior_means = [np.mean(posterior_a), np.mean(posterior_b), np.mean(posterior_c)]
        categories = ['Agricultural and Veterinary Science', 'Business Administration', 'Computer Science']

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(sim_means, posterior_means, s=100, edgecolor='black', alpha=0.7)

        # Annotate points with categories
        for i, category in enumerate(categories):
            plt.annotate(category, (sim_means[i], posterior_means[i]-0.01), fontsize=10, ha='center', va='top')

        # Customize plot appearance
        plt.xlabel('Mean Similarity Score', fontsize=12)
        plt.ylabel('Mean Posterior Probability', fontsize=12)
        plt.title(f'{model_name}', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
        plt.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
        plt.tight_layout()

        output_fp = os.path.join(output_dir, f'{model_name}_sim.png')
        plt.savefig(output_fp, bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":
    exp_prior_fp = "../../0prior/2analysis/normalize_answer.csv"
    exp_posterior_fp = "../../1posterior1/2analysis/normalize_answer.csv"
    exp_similarity_fp = "../../4sim/2analysis/normalize_answer.csv"

    exp_output_dir = "../4plot"
    os.makedirs(exp_output_dir, exist_ok=True)

    main1(exp_prior_fp, exp_posterior_fp, exp_output_dir)
    main2(exp_similarity_fp, exp_posterior_fp, exp_output_dir)

