import os
import pandas as pd


def main(amb_dir, output_dir, list_models, list_names):
    combine_df = pd.DataFrame()
    for name in list_names:
        csv_fp = os.path.join(amb_dir, name, "2analysis", "prob_rotation.csv")
        df = pd.read_csv(csv_fp)
        if combine_df.empty:
            combine_df = df
        else:
            combine_df = pd.merge(combine_df, df, on=["model_name", "rotation"], how="left")
    combine_df = combine_df[combine_df['model_name'].isin(list_models)]
    combine_df = combine_df.drop_duplicates()

    output_fp = os.path.join(output_dir, "1combine.csv")
    combine_df.to_csv(output_fp, index=False)

    combine_df2 = combine_df.copy()
    columns_to_form = combine_df2.columns.difference(["model_name", "rotation"])
    combine_df2[columns_to_form] = combine_df[columns_to_form].round(2)
    output_fp = os.path.join(output_dir, "1combine_present.csv")
    combine_df2.to_csv(output_fp, index=False)


if __name__ == "__main__":
    exp_amb_dir = "../.."
    exp_output_dir = "../1combine"
    os.makedirs(exp_output_dir, exist_ok=True)
    exp_list_names = ["0prior", "1posterior1", "5posterior2", "6posterior3", "7posterior4", "8posterior5",
                      "2inverse1", "3inverse2", "4sim"]
    exp_list_models = ["gpt-4o", "gpt-4-turbo",
                       "claude-3-opus-20240229", "gemini-1.5-pro-latest",
                       "mistral-large-latest"]
    main(exp_amb_dir, exp_output_dir, exp_list_models, exp_list_names)
