import os
import glob
import json
import pandas as pd


def main(raw_dir, list_models, list_names, analysis_dir):
    list_outs = []
    for model_name in list_models:
        for name in list_names:
            list_fps = glob.glob(os.path.join(raw_dir, name, model_name, "*.json"))
            for fp in list_fps:
                with open(fp, "rt") as f:
                    data = json.load(f)
                prob = data.get("probability")
                out = {"model_name": model_name, f"{name}": prob}
                list_outs.append(out)

    df = pd.DataFrame(list_outs)
    output_fp = os.path.join(analysis_dir, "normalize_answer.csv")
    df.to_csv(output_fp, index=False)
    print("Done!")


if __name__ == "__main__":
    exp_raw_dir = "../1raw"
    exp_analysis_dir = "../2analysis"
    os.makedirs(exp_analysis_dir, exist_ok=True)
    exp_list_models = ["gpt-4o", "gpt-4-turbo",
                       "gemini-1.5-pro-latest",
                       "claude-3-opus-20240229",
                       "llama-3-70b-instruct",
                       "mistral-large-latest"]
    exp_list_names = ["cs_high", "cs_low", "human_high", "human_low", "unchar_high", "unchar_low"]

    main(exp_raw_dir, exp_list_models, exp_list_names, exp_analysis_dir)
