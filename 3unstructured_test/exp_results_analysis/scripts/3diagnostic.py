import os
import pandas as pd
import numpy as np


def main(posterior_fp, output_dir, list_models, list_rotations):
    df = pd.read_csv(posterior_fp)
    out_df = pd.DataFrame()
    for model_name in list_models:
        model_df = df[df['model_name'] == model_name]
        temp_df = pd.DataFrame()
        temp_df["rotation"] = list_rotations
        temp_df["model_name"] = [model_name] * len(list_rotations)
        for field in ["A", "B", "C"]:
            list1 = model_df[f"p(E|{field})"].tolist()
            list2 = model_df[f"p(E|-{field})"].tolist()
            list_diagnostics = np.array(list1) / np.array(list2)
            temp_df[f"D(E,{field})"] = list_diagnostics
        out_df = pd.concat([out_df, temp_df], ignore_index=True)

    out_df = out_df[["model_name", "rotation", "D(E,A)", "D(E,B)", "D(E,C)"]]

    output_fp = os.path.join(output_dir, "3diagnostic.csv")
    out_df.to_csv(output_fp, index=False)

    out_df2 = out_df.copy()
    out_df2[["D(E,A)", "D(E,B)", "D(E,C)"]] = out_df2[["D(E,A)", "D(E,B)", "D(E,C)"]].round(2)
    output_fp = os.path.join(output_dir, "3diagnostic_present.csv")
    out_df2.to_csv(output_fp, index=False)


if __name__ == "__main__":
    exp_posterior_fp = "../1combine/1combine.csv"
    exp_output_dir = "../3diagnostic"
    os.makedirs(exp_output_dir, exist_ok=True)
    exp_list_models = ["gpt-4o", "gpt-4-turbo",
                       "claude-3-opus-20240229", "gemini-1.5-pro-latest",
                       "mistral-large-latest"]
    exp_list_rotations = ["A", "B", "C", "A_B", "A_C", "B_C", "A_B_C"]
    main(exp_posterior_fp, exp_output_dir, exp_list_models, exp_list_rotations)
