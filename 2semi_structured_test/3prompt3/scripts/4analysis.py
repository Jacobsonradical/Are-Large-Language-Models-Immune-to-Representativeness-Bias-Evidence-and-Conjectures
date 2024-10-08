import os
import pandas as pd
import numpy as np


def main(norm_fp, list_names, analysis_dir):
    list_outs = []
    norm_df = pd.read_csv(norm_fp)
    for model_name in norm_df["model_name"].unique():
        df = norm_df[norm_df["model_name"] == model_name]
        for name in list_names:
            list_high_probs = df[f"{name}_high"].dropna().tolist()
            list_low_probs = df[f"{name}_low"].dropna().tolist()
            high_mean = np.mean(list_high_probs)
            low_mean = np.mean(list_low_probs)
            difference = high_mean - low_mean
            ratio = (high_mean/(1-high_mean))/(low_mean/(1-low_mean))
            out = {
                "model_name": model_name,
                f"{name}_high": high_mean,
                f"{name}_low": low_mean,
                f"{name}_difference": difference,
                f"{name}_ratio": ratio
            }
            list_outs.append(out)

    out_df = pd.DataFrame(list_outs)
    output1_fp = os.path.join(analysis_dir, "prob.csv")
    out_df.to_csv(output1_fp, index=False)

    out_df2 = out_df.copy()
    col_to_form = out_df2.columns.difference(['model_name'])
    out_df2[col_to_form] = out_df[col_to_form].round(2)
    output2_fp = os.path.join(analysis_dir, "prob_present.csv")
    out_df2.to_csv(output2_fp, index=False)

    print("Done!")


if __name__ == "__main__":
    exp_norm_fp = "../2analysis/normalize_answer.csv"
    exp_analysis_dir = "../2analysis"
    os.makedirs(exp_analysis_dir, exist_ok=True)
    exp_list_models = ["gpt-4o"]
    exp_list_names = ["cs", "human", "unchar"]

    main(exp_norm_fp, exp_list_names, exp_analysis_dir)
