import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main1(prob_fp, list_schemes, output_dir):
    df = pd.read_csv(prob_fp)

    # Directory for saving plots
    plot_dir = os.path.join(output_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    # Create and save the baseline figure with just the lines and labels
    plt.figure(figsize=(8, 8))

    # 45-degree line (Representative)
    x_vals = np.linspace(0, 1, 100)
    plt.plot(x_vals, x_vals, '--', label='Representative: y=x', alpha=0.7, color="#008080", linewidth=2)

    # Line for y/(1-y) = 9x/(1-x) (Normative)
    y_vals = (9 * x_vals) / (8 * x_vals + 1)
    plt.plot(x_vals, y_vals, ':', label='Normative: y/(1-y) = 9x/(1-x)', alpha=1, color="#008080", linewidth=2.5)

    # Label axes
    plt.xlabel("Posterior probability in low base rate setting")
    plt.ylabel("Posterior probability in high base rate setting")
    plt.title("Baseline Plot with Normative Curve and Representative Line")

    # Legend
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    # Save the baseline plot
    baseline_fp = os.path.join(plot_dir, "baseline_plot.png")
    plt.savefig(baseline_fp, bbox_inches='tight', dpi=300)
    plt.close()

    # Define updated color scheme
    scheme_styles = {
        'unchar': {'color': '#A52A2A', 'marker': 'o', 'label': 'Uncharacteristic Description'},
        'cs': {'color': '#800080', 'marker': '^', 'label': 'Computer Science Description'},
        'human': {'color': '#008080', 'marker': 's', 'label': 'Humanity Description'},
    }

    # Create a subplot for all model-specific graphs
    model_names = df["model_name"].unique()
    n_models = len(model_names)
    cols = 2  # Number of columns in the subplot grid
    rows = -(-n_models // cols)  # Calculate rows needed (ceiling division)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = axes.flatten()

    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        df1 = df[df["model_name"] == model_name]

        # 45-degree line (Representative)
        ax.plot(x_vals, x_vals, '--', label='Representative: y=x', alpha=0.7, color="#b3b300", linewidth=2)

        # Line for y/(1-y) = 9x/(1-x) (Normative)
        ax.plot(x_vals, y_vals, ':', label='Normative: y/(1-y) = 9x/(1-x)', alpha=1, color="#b3b300", linewidth=2.5)

        for high, low in list_schemes:
            y = df1[high].dropna().iloc[0]
            x = df1[low].dropna().iloc[0]
            # Determine style based on scheme
            scheme = next((key for key in scheme_styles if key in high or key in low), None)
            if scheme:
                style = scheme_styles[scheme]
                ax.scatter(x, y, color=style['color'], marker=style['marker'],
                           alpha=1, label=style['label'], s=100)

        # Label axes
        ax.set_xlabel("Posterior probability in low base rate setting")
        ax.set_ylabel("Posterior probability in high base rate setting")
        ax.set_title(f"{model_name}", fontweight='bold', fontsize=14)

        # Legend (handles unique labels only)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)

        # Make square axis
        ax.axis('square')

    # Remove unused subplots
    for ax in axes[n_models:]:
        fig.delaxes(ax)

    # Adjust layout and save the combined plot
    plt.tight_layout()
    combined_fp = os.path.join(plot_dir, "combined_models.png")
    plt.savefig(combined_fp, bbox_inches='tight', dpi=300)
    plt.close()


def main2(prob_fp, list_schemes, output_dir):
    df = pd.read_csv(prob_fp)

    # Directory for saving plots
    plot_dir = os.path.join(output_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    # Create and save the figure with just the lines and labels
    plt.figure(figsize=(8, 8))

    # 45-degree line (Representative)
    x_vals = np.linspace(0, 1, 100)
    plt.plot(x_vals, x_vals, '--', label='Representative: y=x', alpha=0.7, color="#008080", linewidth=2)

    # Line for y/(1-y) = 9x/(1-x) (Normative)
    y_vals = (9 * x_vals) / (8 * x_vals + 1)
    plt.plot(x_vals, y_vals, ':', label='Normative: y/(1-y) = 9x/(1-x)', alpha=1, color="#008080", linewidth=2.5)

    # Label axes
    plt.xlabel("Posterior probability in low base rate setting")
    plt.ylabel("Posterior probability in high base rate setting")
    plt.title("Baseline Plot with Normative Curve and Representative Line")

    # Legend
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    # Save the baseline plot
    baseline_fp = os.path.join(plot_dir, "baseline_plot.png")
    plt.savefig(baseline_fp, bbox_inches='tight', dpi=300)
    plt.close()

    # Define updated color scheme
    scheme_styles = {
        'unchar': {'color': '#A52A2A', 'marker': 'o', 'label': 'Uncharacteristic Description'},
        'cs': {'color': '#800080', 'marker': '^', 'label': 'Computer Science Description'},
        'human': {'color': '#008080', 'marker': 's', 'label': 'Humanity Description'},
    }

    # Continue with the main plotting for each model
    for model_name in df["model_name"].unique():
        df1 = df[df["model_name"] == model_name]

        plt.figure(figsize=(8, 8))

        # 45-degree line (Representative)
        plt.plot(x_vals, x_vals, '--', label='Representative: y=x', alpha=0.7, color="#b3b300", linewidth=2)

        # Line for y/(1-y) = 9x/(1-x) (Normative)
        plt.plot(x_vals, y_vals, ':', label='Normative: y/(1-y) = 9x/(1-x)', alpha=1, color="#b3b300", linewidth=2.5)

        for high, low in list_schemes:
            y = df1[high].dropna().iloc[0]
            x = df1[low].dropna().iloc[0]
            # Determine style based on scheme
            scheme = next((key for key in scheme_styles if key in high or key in low), None)
            if scheme:
                style = scheme_styles[scheme]
                plt.scatter(x, y, color=style['color'], marker=style['marker'],
                            alpha=1, label=style['label'], s=100)

        # Label axes
        plt.xlabel("Posterior probability in low base rate setting")
        plt.ylabel("Posterior probability in high base rate setting")
        plt.title(f"{model_name}", fontsize=14, fontweight='bold')

        # Legend (handles unique labels only)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

        # Show plot
        plt.axis('square')
        plt.tight_layout()
        output_fp = os.path.join(plot_dir, f"{model_name}.png")
        plt.savefig(output_fp, bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":
    exp_prob_fp = "../2analysis/prob.csv"

    exp_output_dir = "../2analysis"

    exp_list_schemes = [("cs_high", "cs_low"), ("human_high", "human_low"), ("unchar_high", "unchar_low")]

    main1(exp_prob_fp, exp_list_schemes, exp_output_dir)
    main2(exp_prob_fp, exp_list_schemes, exp_output_dir)
