import os
import glob
import pandas as pd
from collections import defaultdict
import numpy as np


def main(norm_dir, output_dir, model_dict):
    list_fps = glob.glob(os.path.join(norm_dir, '*.csv'))
    for model_size, list_models, in model_dict.items():
        out_dict = defaultdict(list)
        max_len = 0

        for fp in list_fps:
            test_name = os.path.basename(fp).split(".")[0]
            df = pd.read_csv(fp)
            for model_name in list_models:
                model_df = df[df['model_name'] == model_name]
                list_answers = model_df["answer"].tolist()
                out_dict[f"{test_name}"].extend(list_answers)
                if len(out_dict[test_name]) > max_len:
                    max_len = len(out_dict[test_name])

        for key in out_dict:
            if len(out_dict[key]) < max_len:
                out_dict[key].extend([np.nan] * (max_len - len(out_dict[key])))

        out_df = pd.DataFrame(out_dict)
        output_fp = os.path.join(output_dir, f"{model_size}.csv")
        out_df.to_csv(output_fp, index=False)
        print(f"Done for {model_size}!")

    print("All done!")


if __name__ == "__main__":
    exp_norm_dir = "../2analysis/1normalize_answer"
    exp_output_dir = "../2analysis/2normalize_answer_group"
    os.makedirs(exp_output_dir, exist_ok=True)

    exp_model_dict = dict(
        big=["gpt-4o", "gpt-4-turbo", "gemini-1.5-pro-latest", "claude-3-opus-20240229", "mistral-large-latest"],
        mid=["gemini-1.5-flash-latest", "claude-3-sonnet-20240229", "open-mixtral-8x22b", "mistral-medium-latest"],
        small=["gpt-3.5-turbo-0125", "gemini-1.0-pro-latest", "claude-3-haiku-20240307", "open-mixtral-8x7b",
               "open-mistral-7b", "mistral-small-latest"])

    main(exp_norm_dir, exp_output_dir, exp_model_dict)
