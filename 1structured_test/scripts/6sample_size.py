import os
import glob
import json
import pandas as pd
import scipy.stats as stats
import numpy as np


def get_accuracy(group_norm_dir, answer_dict, output_dir):
    accuracy_dir = os.path.join(output_dir, "accumulative-accuracy-rate")
    os.makedirs(accuracy_dir, exist_ok=True)
    list_fps = glob.glob(os.path.join(group_norm_dir, '*.csv'))
    for fp in list_fps:
        list_outs = []
        exp_name = os.path.basename(fp).split(".csv")[0]
        set_correct_answers = answer_dict[exp_name]
        df = pd.read_csv(fp)
        for model_name in df["model_name"].unique():
            df1 = df[df["model_name"] == model_name]
            list_answers = df1["answer"].tolist()
            total = 0
            right = 0
            for answer in list_answers:
                if answer in set_correct_answers:
                    right += 1
                total += 1
                accuracy_rate = right / total
                out = {"model_name": model_name, "accuracy_rate": accuracy_rate}
                list_outs.append(out)
        out_df = pd.DataFrame(list_outs)
        output_fp = os.path.join(accuracy_dir, exp_name + ".csv")
        out_df.to_csv(output_fp, index=False)


def get_sample_size(alpha, tolerance, output_dir):
    list_fps = glob.glob(os.path.join(output_dir, "accumulative-accuracy-rate", "*.csv"))
    for fp in list_fps:
        list_outs = []
        exp_name = os.path.basename(fp).split(".csv")[0]
        df = pd.read_csv(fp)
        for model_name in df["model_name"].unique():
            df1 = df[df["model_name"] == model_name]
            list_rates = df1["accuracy_rate"].tolist()
            current_sample = len(list_rates)
            degree = 1
            chi_square = stats.chi2.ppf(1 - alpha, degree)
            variance = np.var(list_rates)
            required_sample = (4 * chi_square * variance) / (tolerance ** 2)
            resample = 1 if required_sample > current_sample else 0
            out = {"model_name": model_name,
                   "variance": variance,
                   "current_sample": current_sample,
                   "required_sample": required_sample,
                   "resample": resample
                   }
            list_outs.append(out)
        out_df = pd.DataFrame(list_outs)
        output_fp = os.path.join(output_dir, exp_name + ".csv")
        out_df.to_csv(output_fp, index=False)


if __name__ == "__main__":
    exp_group_norm_dir = "../2analysis/1normalize_answer"
    exp_output_dir = "../2analysis/6sample_size"
    os.makedirs(exp_output_dir, exist_ok=True)

    exp_alpha = 0.01
    exp_tolerance = 0.05

    exp_answer_dict = {"1birth": {45, 45.0, "45", "45.0"},
                       "2card": {4, 4.0, "4", "4.0"},
                       "3balls": {1, 1.0, "1", "1.0"},
                       "4hospital": {2, 2.0, "2", "2.0"},
                       "5investigator": {2, 2.0, "2", "2.0"},
                       "6team": {2, 2.0, "2", "2.0"},
                       "7attacker": (0.40, 0.43),
                       "8attacker": (0.40, 0.43),
                       "9attacker": {0.2, 0.20, "0.2", "0.20"},
                       "10attacker": (0.40, 0.43),
                       "11attacker": (0.40, 0.43),
                       "12attacker": (0.40, 0.43),
                       "13dreamer": {0.2, 0.20, "0.2", "0.20"},
                       "14dreamer": {0.2, 0.20, "0.2", "0.20"},
                       "15dreamer": {0.2, 0.20, "0.2", "0.20"},
                       "16dreamer": {0.2, 0.20, "0.2", "0.20"}}

    get_accuracy(exp_group_norm_dir, exp_answer_dict, exp_output_dir)
    get_sample_size(exp_alpha, exp_tolerance, exp_output_dir)
