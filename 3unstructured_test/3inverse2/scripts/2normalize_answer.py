import os
import glob
import json
import pandas as pd

name_dict = {"A": "agricultural and veterinary science",
             "C": "computer science",
             "B": "business administration"}


def main(raw_dir, list_models, list_names, analysis_dir):
    list_outs = []
    for model_name in list_models:
        for name in list_names:
            list_comps = name.split("_")
            list_fps = glob.glob(os.path.join(raw_dir, name, model_name, "*.json"))
            for fp in list_fps:
                with open(fp, "rt") as f:
                    data = json.load(f)
                out = {"model_name": model_name, "rotation": name}
                for comp in list_comps:
                    prob = data.get(name_dict.get(comp))
                    out[f"p(E|-{comp})"] = prob
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
                       "claude-3-opus-20240229", "gemini-1.5-pro-latest",
                       "mistral-large-latest"]
    exp_list_names = ["A", "B", "C", "A_B", "A_C", "B_C", "A_B_C"]

    main(exp_raw_dir, exp_list_models, exp_list_names, exp_analysis_dir)
