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
import random

client_GPT = OpenAI(api_key="ENTER YOUR KEY")  # OPENAI API
genai.configure(api_key="ENTER YOUR KEY")  # GEMINI API
client_CLAUDE = anthropic.Anthropic(api_key="ENTER YOUR KEY")  # CLAUDE API
client_MIS = MistralClient(api_key='ENTER YOUR KEY')  # MISTRAL API


def make_prompt_and_record_order(begin, choice, end):
    first_two = choice[0:2]
    random.shuffle(first_two)
    prompt_dict = {"a": first_two[0], "b": first_two[1]}

    order_dict = {"a": 1 if first_two[0] == choice[0] else 2,
                  "b": 2 if first_two[1] == choice[1] else 1}

    final_prompt = begin + "\n\n" + json.dumps(prompt_dict, indent=2) + "\n\n" + end
    return final_prompt, order_dict


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
        if norm_key != "probability":
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
        if not 0 <= value <= 1:
            print(f"{model_name}. Value not between 0 and 1. Retry test round {test_round}. Attempt {attempt}.")
            return False
        norm_subdict[norm_key] = value
    with atomic_write(output_filepath, overwrite=True) as f:
        json.dump(norm_subdict, f)
    print(f"Done for round {test_round} for {model_name} with attempt {attempt}.")
    return True


def exp(amb_dir, begin, choice, end, model_name, starting_round, ending_round, max_tokens):
    output_dir = os.path.join(amb_dir, f"{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    response_dir = os.path.join(output_dir, "response")
    os.makedirs(response_dir, exist_ok=True)
    order_dir = os.path.join(output_dir, "order")
    os.makedirs(order_dir, exist_ok=True)
    prompt_dir = os.path.join(output_dir, "prompt")
    os.makedirs(prompt_dir, exist_ok=True)

    for test_round in range(starting_round, ending_round + 1):
        output_filepath = os.path.join(output_dir, f"{model_name}_{test_round}.json")
        if os.path.exists(output_filepath):
            print(f"Already done for {model_name}_{test_round}. Skip.")
            continue

        final_prompt, order_dict = make_prompt_and_record_order(begin, choice, end)
        with open(os.path.join(prompt_dir, f"{model_name}_{test_round}.txt"), "w") as f:
            f.write(final_prompt)
        with open(os.path.join(order_dir, f"{model_name}_{test_round}.json"), "w") as f:
            json.dump(order_dict, f)

        attempt = 1
        while True:
            try:
                result_dict = call_api(final_prompt, model_name, max_tokens)
            except Exception as e:
                print(f"response: {e}")
                return
            try:
                if handle_reply(result_dict, model_name, test_round, attempt, output_filepath, response_dir):
                    break
            except Exception as e:
                print(f"handle reply: {model_name}, {e}")

            attempt += 1
            if attempt == 5:
                break


if __name__ == "__main__":
    exp_amb_dir = "../1raw/11attacker"
    os.makedirs(exp_amb_dir, exist_ok=True)
    with open("11attacker_begin.txt", "r") as f:
        exp_begin = f.read()
    with open("11attacker_choice.json", "r") as f:
        exp_choice = json.load(f)
    with open("11attacker_end.txt", "r") as f:
        exp_end = f.read()

    # gemini-1.5-pro-latest refused to answer the question
    model_round = {"gpt-4o": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "gpt-4-turbo": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "gpt-3.5-turbo-0125": [(i, i + 9) for i in range(1, 281, 9 + 1)],
                   # "gemini-1.5-pro-latest": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "gemini-1.5-flash-latest": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "gemini-1.0-pro-latest": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "claude-3-opus-20240229": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "claude-3-sonnet-20240229": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "claude-3-haiku-20240307": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "mistral-large-latest": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "mistral-medium-latest": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                   "mistral-small-latest": [(i, i + 9) for i in range(1, 171, 9 + 1)]}
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
        futures = [
            executor.submit(exp, exp_amb_dir, exp_begin, exp_choice, exp_end, exp_model_name, start, end, 2000)
            for exp_model_name, ranges in model_round.items()
            for start, end in ranges
        ]

    #  open-mixtral-8x22b, open-mixtral-8x7b and open-mistral-7b need special design of end_prompt for parsable answer.
    model_round_2 = {"open-mixtral-8x22b": [(i, i + 9) for i in range(1, 151, 9 + 1)],
                     "open-mixtral-8x7b": [(i, i + 9) for i in range(1, 351, 9 + 1)],
                     "open-mistral-7b": [(i, i + 9) for i in range(1, 1001, 9 + 1)]}
    with open("11attacker_end-special.txt", "r") as f:
        exp_end_2 = f.read()
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
        futures_2 = [
            executor.submit(exp, exp_amb_dir, exp_begin, exp_choice, exp_end_2, exp_model_name_2, start, end, 2000)
            for exp_model_name_2, ranges_2 in model_round_2.items()
            for start, end in ranges_2
        ]
