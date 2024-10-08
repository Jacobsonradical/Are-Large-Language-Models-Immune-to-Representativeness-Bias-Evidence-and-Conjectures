import concurrent.futures
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
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
client_MIS = MistralClient(api_key="ENTER YOUR KEY")  # MISTRAL API

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
                                      safety_settings={HarmCategory.HARM_CATEGORY_HATE_SPEECH:
                                                           HarmBlockThreshold.BLOCK_NONE,
                                                       HarmCategory.HARM_CATEGORY_HARASSMENT:
                                                           HarmBlockThreshold.BLOCK_NONE,
                                                       HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
                                                           HarmBlockThreshold.BLOCK_NONE,
                                                       HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                                                           HarmBlockThreshold.BLOCK_NONE},
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


def handle_reply(result_dict, model_name, test_round, attempt, output_filepath, response_dir, list_fields, name):
    response_filepath = os.path.join(response_dir, f"{model_name}_{test_round}_{attempt}.txt")
    with open(response_filepath, "w") as file:
        file.write(result_dict)

    # Find the starting index of the LAST dictionary in the string.
    # This is in case of more than 1 dictionary.
    dict_start = result_dict.rfind('{')

    if dict_start == -1:
        print(f"{model_name}, {name}. No dictionary found. Retry text round {test_round}. Attempt {attempt}.")
        return False
    else:
        try:
            broken_dict = result_dict[dict_start:]
            fixed_dict = repair_json(broken_dict)
            fixed_data = json.loads(fixed_dict)
        except:
            print(f"{model_name}, {name}. Json Repair Fail. Retry test round {test_round}. Attempt {attempt}.")
            return False

    normalized_subdict = {str(key).lower().strip(): value for key, value in fixed_data.items()}
    for field in list_fields:
        if field not in normalized_subdict:
            print(f"{model_name}, {name}. Not Enough Key. Retry test round {test_round}. Attempt {attempt}.")
            return False

    if len(list(normalized_subdict.keys())) != len(list_fields):
        print(f"{model_name}, {name}. Wrong number of keys. Retry test round {test_round}. Attempt {attempt}.")
        return False

    norm_subdict = {}
    for key, value in normalized_subdict.items():
        norm_key = str(key).lower().strip()
        if norm_key not in list_fields:
            print(f"{model_name}, {name}. Corrupted Key. Retry test round {test_round}. Attempt {attempt}.")
            return False
        else:
            try:
                str_value = str(value).strip().replace('%', '')
                norm_value = float(str_value)
                if not 0 <= norm_value <= 1:
                    print(f"{model_name}, {name}. Bad Answers. Retry test round {test_round}. Attempt {attempt}.")
                    return False
            except:
                print(f"{model_name}, {name}. Corrupted answer. Retry test round {test_round}. Attempt {attempt}.")
                return False
        norm_subdict[norm_key] = norm_value
    sorted_norm_subdict = {field: norm_subdict[field] for field in list_fields}
    with atomic_write(output_filepath, overwrite=True) as f:
        json.dump(sorted_norm_subdict, f)
    print(f"{model_name}, {name}. Done for round {test_round} for {model_name} with attempt {attempt}.")
    return True


def exp(amb_dir, name, prompt, model_name, starting_round, ending_round, max_tokens):
    output_dir = os.path.join(amb_dir, f"{model_name}")  # output directory for each model
    os.makedirs(output_dir, exist_ok=True)
    response_dir = os.path.join(output_dir, "response")  # response txt directory for each model
    os.makedirs(response_dir, exist_ok=True)

    for test_round in range(starting_round, ending_round + 1):
        output_filepath = os.path.join(output_dir, f"{model_name}_{test_round}.json")
        if os.path.exists(output_filepath):
            print(f"{model_name}, {name}. Already done for {model_name}_{test_round}. Skip.")
            continue

        list_fields = ["probability"]

        attempt = 1
        while True:
            try:
                result_dict = call_api(prompt, model_name, max_tokens)
            except Exception as e:
                print(f"{model_name}, {name}. response: {e}")
                return
            try:
                if handle_reply(result_dict, model_name, test_round, attempt,
                                output_filepath, response_dir, list_fields, name):
                    break
            except Exception as e:
                print(f"{model_name}, {name}. handle reply: {model_name}, {e}")

            attempt += 1
            if attempt == 5:
                break


if __name__ == "__main__":
    list_names = ["cs_high", "cs_low", "human_high", "human_low", "unchar_high", "unchar_low"]

    with open("model_round.json", "rt") as f:
        model_round = json.load(f)

    for exp_name in list_names:
        exp_amb_dir = f"../1raw/{exp_name}"
        os.makedirs(exp_amb_dir, exist_ok=True)
        with open(f"{exp_name}.txt", "rt") as f:
            exp_prompt = f.read()
        with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
            futures = [
                executor.submit(exp, exp_amb_dir, exp_name, exp_prompt, exp_model_name, start, end, max_tokens=2000)
                for exp_model_name, sub_dict in model_round.items()
                for start, end in [(i, i + 4) for i in range(1, sub_dict.get(exp_name), 4 + 1)]
            ]
