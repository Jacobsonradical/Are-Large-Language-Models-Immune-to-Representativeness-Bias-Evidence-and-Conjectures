import gc
import transformers
import torch
import json
import os
from atomicwrites import atomic_write
from json_repair import repair_json
from huggingface_hub import login
from datasets import Dataset
import pandas as pd
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset

login("ENTER YOUR HUGGINGFACT HUB API KEY")  # You also have to request for Llama3


def handle_reply(result_dict, list_choices, model_name, test_round, attempt, output_filepath, debug_dir, response_dir):
    response_filepath = os.path.join(response_dir, f"{model_name}_{test_round}_{attempt}.txt")
    with open(response_filepath, "w") as file:
        file.write(result_dict)

    dict_start = result_dict.rfind('{')  # Find the starting index of the LAST dictionary in the string. This is in
    # case of more than 1 dictionary.

    if dict_start == -1:
        print(f"{model_name}. No dictionary found. Retry text round {test_round}. Attempt {attempt}.")
        return
    else:
        try:
            borken_dict = result_dict[dict_start:]
            fixed_dict = repair_json(borken_dict)
            fixed_data = json.loads(fixed_dict)
        except:
            print(f"{model_name}. Json Repair Fail. Retry test round {test_round}. Attempt {attempt}.")
            return

    normalized_subdict = {str(key).lower().strip(): value for key, value in fixed_data.items()}
    for field in list_choices:
        if field not in normalized_subdict:
            print(f"{model_name}. Not Enough Key. Retry test round {test_round}. Attempt {attempt}.")
            return

    norm_subdict = {}
    for key, value in fixed_data.items():
        norm_key = str(key).lower().strip()
        if norm_key == "explanation":
            continue
        elif norm_key not in list_choices:
            print(f"{model_name}. Corrupted Key. Retry test round {test_round}. Attempt {attempt}.")
            return
        if isinstance(value, str):
            try:
                value = float(value)
            except:
                print(
                    f"{model_name}. String value cannot change to float. Retry test round {test_round}. Attempt {attempt}.")
                return
        elif not isinstance(value, float) and not isinstance(value, int):
            print(f'{model_name}. Non-float value. Retry test round {test_round}. Attempt {attempt}.')
            return
        if not 0 <= value <= 1:
            print(f"{model_name}. Value not between 0 and 1. Retry test round {test_round}. Attempt {attempt}.")
            return
        norm_subdict[norm_key] = value
    sorted_norm_subdict = {key: norm_subdict[key] for key in list_choices}
    with atomic_write(output_filepath, overwrite=True) as f:
        json.dump(sorted_norm_subdict, f)
    print(f"Done for round {test_round} for {model_name} with attempt {attempt}.")


def exp(amb_dir, list_choices, prompt, model_name, starting_round, ending_round, attempt, batch_size=200,
        max_tokens=350):
    output_dir = os.path.join(amb_dir, f"{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    response_dir = os.path.join(output_dir, "response")
    os.makedirs(response_dir, exist_ok=True)

    prompt_dir = os.path.join(output_dir, "prompt")
    os.makedirs(prompt_dir, exist_ok=True)

    set_rounds = set(range(starting_round, ending_round + 1))
    for i in range(starting_round, ending_round + 1):
        output_filepath = os.path.join(output_dir, f"{model_name}_{i}.json")
        if os.path.exists(output_filepath):
            print(f"Already done for {model_name}_{i}. Skip.")
            set_rounds.remove(i)
    if not set_rounds:
        print("skip all.")
        return

    data = []
    for i in tqdm(set_rounds, desc="Generating Dataset"):
        prompt_fp = os.path.join(prompt_dir, f"{model_name}_{i}.txt")
        with atomic_write(prompt_fp, overwrite=True) as f:
            f.write(prompt)

        messages = [{"role": "user", "content": prompt}, ]
        data.append({"round": i,
                     "model_name": model_name,
                     "messages": messages})
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    dataset.save_to_disk(
        os.path.join(debug_dir, f"{model_name}_dataset_{starting_round}_{ending_round}_{len(set_rounds)}"))
    print("Dataset Generated and Saved!")

    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    for device_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()

    gc.collect()
    print("GPU memory cleared for all GPUs")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

    terminators = [pipeline.tokenizer.eos_token_id,
                   pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for index, output in enumerate(pipeline(KeyDataset(dataset, "messages"),
                                            batch_size=batch_size,
                                            max_new_tokens=max_tokens,
                                            eos_token_id=terminators,
                                            do_sample=True,
                                            temperature=0.6,
                                            top_p=0.9, )):
        result_dict = output[0]["generated_text"][-1].get('content')
        list_rounds = list(set_rounds)
        test_round = list_rounds[index]
        output_filepath = os.path.join(output_dir, f"{model_name}_{test_round}.json")
        handle_reply(result_dict, list_choices, model_name, test_round, attempt, output_filepath, debug_dir,
                     response_dir)


if __name__ == "__main__":
    exp_list_choices = ["probability"]
    list_names = ["cs_high", "cs_low", "human_high", "human_low", "unchar_high", "unchar_low"]

    # CUDA 12.2, 4 x H100 SXM, ~ 320GB
    for name in list_names:
        exp_amb_dir = f"../1raw/{name}"
        os.makedirs(exp_amb_dir, exist_ok=True)
        with open(f"{name}.txt", "rt") as f:
            exp_prompt = f.read()
        exp_model_name = "llama-3-70b-instruct"
        for exp_attempt in range(1, 20 + 1):
            exp(exp_amb_dir, exp_list_choices, exp_prompt, exp_model_name, 1, 300,
                exp_attempt, batch_size=200, max_tokens=350)
