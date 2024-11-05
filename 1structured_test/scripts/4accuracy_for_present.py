import os
import glob
import pandas as pd
import sys
from collections import defaultdict


def main(group_norm_dir, output_dir, answer_dict):
    list_fps = glob.glob(os.path.join(group_norm_dir, '*.csv'))
    list_outs = defaultdict(dict)
    for fp in list_fps:
        test_name = os.path.basename(fp).split(".")[0]
        correct_answers = answer_dict.get(f"{test_name}")
        df = pd.read_csv(fp)

        if test_name not in {"7attacker", "8attacker", "10attacker", "11attacker", "12attacker"}:
            for model_name in df["model_name"].unique():
                model_df = df[df["model_name"] == model_name]
                list_answers = model_df["answer"].dropna().tolist()
                correct_count = 0
                for answer in list_answers:
                    if answer in correct_answers:
                        correct_count += 1
                correct_rate = correct_count / len(list_answers)
                list_outs[model_name][test_name] = correct_rate
                print(f"{model_name}, {test_name}: Done")

        else:
            for model_name in df["model_name"].unique():
                model_df = df[df["model_name"] == model_name]
                list_answers = model_df["answer"].dropna().tolist()
                list_answer1 = [i for i in list_answers if i != 'True']
                if len(list_answer1) != len(list_answers):
                    print(f"{test_name}, {model_name}. Drop {len(list_answers) - len(list_answer1)} due to some unknown "
                          f"reason of answers with value = True")
                list_answer1 = [float(i) for i in list_answer1]

                correct_count = 0
                for answer in list_answer1:
                    if correct_answers[0] <= answer <= correct_answers[1]:
                        correct_count += 1
                correct_rate = correct_count / len(list_answer1)
                list_outs[model_name][test_name] = correct_rate
                print(f"{model_name}, {test_name}: Done")

        df2 = pd.DataFrame(list_outs)
        output_fp = os.path.join(output_dir, 'accuracy_rate.csv')
        df2.to_csv(output_fp, index=False)

        df3 = df2.copy()
        df3 = df3.round(2).T
        output_fp = os.path.join(output_dir, 'accuracy_rate_present.csv')
        df3.to_csv(output_fp, index=True)

    print("All done!")


if __name__ == "__main__":
    exp_group_norm_dir = "../2analysis/1normalize_answer"
    exp_output_dir = "../2analysis/4accuracy_for_present"
    os.makedirs(exp_output_dir, exist_ok=True)

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

    main(exp_group_norm_dir, exp_output_dir, exp_answer_dict)
