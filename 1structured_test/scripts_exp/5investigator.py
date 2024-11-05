import concurrent.futures
from openai import OpenAI
import google.generativeai as genai
import anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
import json
from atomicwrites import atomic_write
from json_repair import repair_json


client_GPT = OpenAI(api_key="ENTER YOUR KEY")  # OPENAI API
genai.configure(api_key="ENTER YOUR KEY")  # GEMINI API
client_CLAUDE = anthropic.Anthropic(api_key="ENTER YOUR KEY")  # CLAUDE API
client_MIS = MistralClient(api_key='ENTER YOUR KEY')  # MISTRAL API


def call_api(prompt, model_name, max_tokens):
    result_dict = None

    if model_name in {"gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o"}:
        test_result = client_GPT.chat.completions.create(model=model_name,
                                                         messages=[{"role": "user",
                                                                    "content": [{"type": "text",
                                                                                 "text": prompt}]}],
                                                         temperature=1,
                                                         max_tokens=max_tokens,
                                                         top_p=1,
                                                         frequency_penalty=0,
                                                         presence_penalty=0)
        result_dict = test_result.choices[0].message.content

    if model_name in {"gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro-latest"}:
        model = genai.GenerativeModel(model_name=f'models/{model_name}',
                                      generation_config={'candidate_count': None,
                                                         'stop_sequences': None,
                                                         'max_output_tokens': max_tokens,
                                                         'temperature': None,
                                                         'top_p': None,
                                                         'top_k': None, },
                                      safety_settings={},
                                      tools=None)
        response = model.generate_content(prompt)
        result_dict = response.text

    if model_name in {"claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"}:
        message = client_CLAUDE.messages.create(model=model_name,
                                                max_tokens=max_tokens,
                                                messages=[{"role": "user",
                                                           "content": [{"type": "text",
                                                                        "text": prompt}]}])
        result_dict = message.content[0].text

    if model_name in {"open-mixtral-8x22b", "open-mixtral-8x7b", "open-mistral-7b", "mistral-large-latest",
                      "mistral-medium-latest", "mistral-small-latest"}:
        response = client_MIS.chat(model=model_name,
                                   messages=[ChatMessage(role="user", content=prompt)],
                                   max_tokens=max_tokens)
        result_dict = response.choices[0].message.content
    return result_dict


def handle_reply(result_dict, model_name, test_round, attempt, output_filepath, response_dir):
    response_filepath = os.path.join(response_dir, f"{model_name}_{test_round}_{attempt}.txt")
    with open(response_filepath, "w") as file:
        file.write(result_dict)

    # Find the starting index of the LAST dictionary in the string.
    # This is in case of more than 1 dictionary.
    dict_start = result_dict.rfind('{')

    if dict_start == -1:
        print(f"{model_name}. No dictionary found. Retry text round {test_round}. Attempt {attempt}.")
        return False
    else:
        try:
            broken_dict = result_dict[dict_start:]
            fixed_dict = repair_json(broken_dict)
            fixed_data = json.loads(fixed_dict)
        except:
            print(f"{model_name}. Json Repair Fail. Retry test round {test_round}. Attempt {attempt}.")
            return False

    if len(fixed_data.items()) != 1:
        print(f"{model_name}. More than one entry. Retry test round {test_round}. Attempt {attempt}.")
        return False

    norm_subdict = {}
    for key, value in fixed_data.items():
        norm_key = str(key).lower().strip()
        if norm_key not in {"investigator"}:
            print(f"{model_name}. Corrupted Key. Retry test round {test_round}. Attempt {attempt}.")
            return False
        if isinstance(value, str):
            try:
                value = float(value)
            except:
                print(f"{model_name}. String value cannot change to float."
                      f"Retry test round {test_round}. Attempt {attempt}.")
                return False
        elif not isinstance(value, float) and not isinstance(value, int):
            print(f'{model_name}. Non-float and non-int value. Retry test round {test_round}. Attempt {attempt}.')
            return False
        if value not in {1, 2, 3, "1", "2", "3"}:
            print(f"{model_name}. Value not between 0 and 1. Retry test round {test_round}. Attempt {attempt}.")
            return False
        norm_subdict[norm_key] = value
    with atomic_write(output_filepath, overwrite=True) as f:
        json.dump(norm_subdict, f)
    print(f"Done for round {test_round} for {model_name} with attempt {attempt}.")
    return True


def exp(amb_dir, prompt, model_name, starting_round, ending_round, max_tokens):
    output_dir = os.path.join(amb_dir, f"{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    response_dir = os.path.join(output_dir, "response")
    os.makedirs(response_dir, exist_ok=True)

    for test_round in range(starting_round, ending_round + 1):
        output_filepath = os.path.join(output_dir, f"{model_name}_{test_round}.json")
        if os.path.exists(output_filepath):
            print(f"Already done for {model_name}_{test_round}. Skip.")
            continue

        attempt = 1
        while True:
            try:
                result_dict = call_api(prompt, model_name, max_tokens)
            except Exception as e:
                print(f"response: {e}")
                return
            try:
                if handle_reply(result_dict, model_name, test_round, attempt, output_filepath, response_dir):
                    break
            except Exception as e:
                print(f"handle reply: {e}.")

            attempt += 1
            if attempt == 5:
                break


if __name__ == "__main__":
    exp_amb_dir = f"../1raw/5investigator"
    os.makedirs(exp_amb_dir, exist_ok=True)

    # open-mistral-7b is excluded due to poor performance
    list_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125",
                   "gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro-latest",
                   "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                   "open-mixtral-8x22b", "open-mixtral-8x7b",
                   "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]

    with open(f"5investigator.txt", "rt") as f:
        exp_prompt = f.read()

    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
        futures = [
            executor.submit(exp, exp_amb_dir, exp_prompt, exp_model_name, start, end, 2000)
            for exp_model_name in list_models
            for start, end in [(i, i + 19) for i in range(1, 100, 19 + 1)]
        ]

    # In a sample of 100, Haiku needs 177
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
        futures_haiku = [
            executor.submit(exp, exp_amb_dir, exp_prompt, "claude-3-haiku-20240307", start, end, 2000)
            for start, end in [(i, i + 19) for i in range(100, 200, 19 + 1)]
        ]
    # In a sample of 100, GPT-3.5 needs 102
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
        futures_gpt35 = [
            executor.submit(exp, exp_amb_dir, exp_prompt, "gpt-3.5-turbo-0125", start, end, 2000)
            for start, end in [(i, i + 19) for i in range(100, 121, 19 + 1)]
        ]
