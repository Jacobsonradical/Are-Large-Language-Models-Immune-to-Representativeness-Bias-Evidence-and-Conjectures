import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from collections import defaultdict
import json


def main(norm_fp, list_names, analysis_dir, alpha, tolerance):
    list_outs = []
    model_round_dict = defaultdict(dict)
    norm_df = pd.read_csv(norm_fp)
    for model_name in norm_df["model_name"].unique():
        df = norm_df[norm_df["model_name"] == model_name]
        for name in list_names:
            df2 = df[df["rotation"] == name]
            list_comps = name.split("_")

            degree = len(list_comps)
            chi_square = stats.chi2.ppf(1 - alpha, degree)
            current_sample = len(df2)

            list_variance = []
            for comp in list_comps:
                list_probs = df2[f'p(E|-{comp})'].tolist()
                variance = np.var(list_probs)
                list_variance.append(variance)
            max_variance = max(list_variance)

            required_sample = (4 * chi_square * max_variance) / (tolerance ** 2)
            resample = 1 if required_sample > current_sample else 0

            out = {"model_name": model_name,
                   "rotation": name,
                   "max_variance": max_variance,
                   "required_sample": required_sample,
                   "current_sample": current_sample,
                   "resample": resample}
            list_outs.append(out)

            model_round_dict[model_name][name] = int((np.ceil(required_sample/10)+2)*10+1) if resample == 1 \
                else int(current_sample+1)

    out_df = pd.DataFrame(list_outs)
    output1_fp = os.path.join(analysis_dir, "sample_size.csv")
    out_df.to_csv(output1_fp, index=False)
    print("Done!")

    out_df1 = out_df.copy()
    out_df1['max_variance'] = out_df1['max_variance'].round(2)
    out_df1['required_sample'] = np.ceil(out_df1['required_sample'])
    output2_fp = os.path.join(analysis_dir, "sample_size_present.csv")
    out_df1.to_csv(output2_fp, index=False)

    output3_fp = os.path.join(analysis_dir, "model_round.json")
    with open(output3_fp, "wt") as f:
        json.dump(model_round_dict, f)


if __name__ == "__main__":
    exp_norm_fp = "../2analysis/normalize_answer.csv"
    exp_analysis_dir = "../2analysis"
    os.makedirs(exp_analysis_dir, exist_ok=True)
    exp_list_names = ["A", "B", "C", "A_B", "A_C", "B_C", "A_B_C"]

    exp_alpha = 0.01
    exp_tolerance = 0.05

    main(exp_norm_fp, exp_list_names, exp_analysis_dir, exp_alpha, exp_tolerance)



