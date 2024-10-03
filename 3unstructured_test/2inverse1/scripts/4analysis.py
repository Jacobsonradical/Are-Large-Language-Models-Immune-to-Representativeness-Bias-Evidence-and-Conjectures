import os
import pandas as pd
import numpy as np


def main_separate_rotation(norm_fp, list_names, analysis_dir):
    list_outs = []
    norm_df = pd.read_csv(norm_fp)
    for model_name in norm_df["model_name"].unique():
        df = norm_df[norm_df["model_name"] == model_name]
        for name in list_names:
            df2 = df[df["rotation"] == name]
            list_comps = name.split("_")
            out = {"model_name": model_name, "rotation": name}
            for comp in list_comps:
                list_probs = df2[f'p(E|{comp})'].tolist()
                mean = np.mean(list_probs)
                out[f'p(E|{comp})'] = mean
            list_outs.append(out)

    out_df = pd.DataFrame(list_outs)
    output1_fp = os.path.join(analysis_dir, "prob_rotation.csv")
    out_df.to_csv(output1_fp, index=False)
    print("Done! Rotation separated.")


def main_combine(norm_fp, short_names, analysis_dir):
    list_outs = []
    norm_df = pd.read_csv(norm_fp)
    for model_name in norm_df["model_name"].unique():
        df = norm_df[norm_df["model_name"] == model_name]
        out = {"model_name": model_name}
        for name in short_names:
            list_probs = df[f'p(E|{name})'].dropna().tolist()
            mean = np.mean(list_probs)
            out[f'p(E|{name})'] = mean
        list_outs.append(out)
    out_df = pd.DataFrame(list_outs)
    output1_fp = os.path.join(analysis_dir, "prob_combine.csv")
    out_df.to_csv(output1_fp, index=False)
    print("Done! Rotation combined.")


if __name__ == "__main__":
    exp_norm_fp = "../2analysis/normalize_answer.csv"
    exp_analysis_dir = "../2analysis"
    os.makedirs(exp_analysis_dir, exist_ok=True)
    exp_list_names = ["A", "B", "C", "A_B", "A_C", "B_C", "A_B_C"]
    exp_short_names = ["A", "B", "C"]
    main_separate_rotation(exp_norm_fp, exp_list_names, exp_analysis_dir)
    main_combine(exp_norm_fp, exp_short_names, exp_analysis_dir)
