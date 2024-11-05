import glob
import os
from collections import defaultdict

import pandas as pd


def main(raw_dir, out_dir):
    list_outs = defaultdict(dict)
    list_folders = os.listdir(raw_dir)
    for folder_name in list_folders:
        list_model_names = os.listdir(os.path.join(raw_dir, folder_name))
        for model_name in list_model_names:
            input_dir = os.path.join(raw_dir, folder_name, model_name)
            list_fps = glob.glob(os.path.join(input_dir, '*.json'))
            sample_size = len(list_fps)
            list_outs[folder_name][model_name] = sample_size

    df = pd.DataFrame(list_outs)
    output_fp = os.path.join(out_dir, 'sample_size.csv')
    df.to_csv(output_fp, index=True)


if __name__ == "__main__":
    exp_raw_dir = "../1raw"
    exp_out_dir = "../2analysis/5sample_size_for_present"
    os.makedirs(exp_out_dir, exist_ok=True)

    main(exp_raw_dir, exp_out_dir)
