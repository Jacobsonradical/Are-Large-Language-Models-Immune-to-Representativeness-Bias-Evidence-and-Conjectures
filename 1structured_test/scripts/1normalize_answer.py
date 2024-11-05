import os
import glob
import json
import pandas as pd


def main(key_dict, raw_dir, output_dir):
    for key, value in key_dict.items():
        input_dir = os.path.join(raw_dir, key)
        list_fps = glob.glob(os.path.join(input_dir, "**", "*.json"))
        list_outs = []
        for fp in list_fps:
            model_name = os.path.basename(os.path.dirname(fp))
            with open(fp, "rt") as f:
                data = json.load(f)
            answer = data.get(value) if value != "2card" else data.get("odds", data.get("odd"))
            out = {"model_name": model_name,
                   "answer": answer}
            list_outs.append(out)
        df = pd.DataFrame(list_outs)
        output_fp = os.path.join(output_dir, f"{key}.csv")
        df.to_csv(output_fp, index=False)
        print(f"Done! {key}.")

    print("All done!")


if __name__ == "__main__":
    exp_key_dict = {"1birth": "answer",
                    "2card": "odds",
                    "3balls": "case",
                    "4hospital": "hospital",
                    "5investigator": "investigator",
                    "6team": "team",
                    "7attacker": "probability",
                    "8attacker": "probability",
                    "9attacker": "probability",
                    "10attacker": "probability",
                    "11attacker": "probability",
                    "12attacker": "probability",
                    "13dreamer": "probability",
                    "14dreamer": "probability",
                    "15dreamer": "probability",
                    "16dreamer": "probability"}

    exp_raw_dir = "../1raw"
    exp_output_dir = "../2analysis/1normalize_answer"
    os.makedirs(exp_output_dir, exist_ok=True)

    main(exp_key_dict, exp_raw_dir, exp_output_dir)
