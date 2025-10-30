# code_release/generate_answers.py

import os
import json
import random
import argparse
import signal
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, wait_random_exponential, stop_after_attempt

try:
    import openai
    import google.generativeai as genai
    from openai import OpenAI
except ImportError:
    openai = None
    genai = None
    OpenAI = None

openrouterclient = None

###################### API KEY AREA ######################
def load_api_keys():
    api_keys = {}
    api_keys_path = os.environ.get("API_KEYS_PATH", "api_keys")
    if os.path.exists(api_keys_path):
        with open(api_keys_path, "r") as f:
            api_keys = json.load(f)
    else:
        # Optionally, load from environment variables
        api_keys["openai"] = os.environ.get("OPENAI_API_KEY")
        api_keys["google"] = os.environ.get("GOOGLE_API_KEY")
        api_keys["openrouter"] = os.environ.get("OPENROUTER_API_KEY")
    return api_keys

api_keys = load_api_keys()
if openai and api_keys.get("openai"):
    openai.api_key = api_keys["openai"]
    openaiclient = OpenAI(api_key=api_keys["openai"])
if genai and api_keys.get("google"):
    genai.configure(api_key=api_keys["google"])
if OpenAI and api_keys.get("openrouter"):
    openrouterclient = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_keys["openrouter"],
    )
##########################################################

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def generate_prompt(row, prompt_method):
    system_prompt = 'You are a group of individuals with these shared characteristics:\n'
    system_prompt += str(row['group_prompt_template'])
    prompt = '**Question**: ' + row['input_template'] + '\n'
    if row.get('group_prompt_variable_map'):
        for variable, value in row['group_prompt_variable_map'].items():
            system_prompt = system_prompt.replace(f'{{{variable}}}', str(value))
    if prompt_method == 'token_prob':
        prompt += 'Do not provide any explanation, only answer with one of the following options: ' + ', '.join(
            row['human_answer'].keys()) + '.\n**Answer**: ('
    elif prompt_method == 'verbalized':
        json_format_str = '{' + ', '.join([f'"{key}": X' for key in row['human_answer'].keys()]) + '}'
        prompt += (
            f'\nEstimate what percentage of your group would choose each option. '
            f'Follow these rules:\n'
            f'1. Use whole numbers from 0 to 100\n'
            f'2. Ensure the percentages sum to exactly 100\n'
            f'3. Only include the numbers (no % symbols)\n'
            f'4. Use this exact valid JSON format: {json_format_str} and do NOT include anything else.\n'
            f'5. Only output your final answer and nothing else. No explanations or intermediate steps are needed.\n'
            f'Replace X with your estimated percentages for each option.\n'
            '**Answer**:'
        )
    return system_prompt, prompt

def get_token_probabilities_local_model(model, tokenizer, prompt, target_tokens):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
        results = {}
        for token in target_tokens:
            encodings = tokenizer.encode(token, add_special_tokens=False)
            # TAYLOR - uhh not sure why this is here - it's not underlining?
            # underscored_encodings = tokenizer.encode("_" + token, add_special_tokens=False)
            probability_sum = sum(probabilities[encoding].item() for encoding in encodings)
            # probability_sum += sum(probabilities[encoding].item() for encoding in underscored_encodings)
            results[token] = probability_sum
    return results

def handler(signum, frame):
    raise TimeoutError

def parse_and_normalize_json_response(response, expected_keys):
    try:
        match = re.search(r'\{[\s\S]*?\}', response)
        if match:
            json_str = match.group(0)
            try:
                response_parsed = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = re.sub(r'%', '', json_str)
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                json_str = re.sub(r'([{,]\s*)([^"\s]+)\s*:', r'\1"\2":', json_str)
                json_str = re.sub(r':\s*([^",}\]\s]+)([,}\]])', r':"\1"\2', json_str)
                json_str = re.sub(r':"(-?\d+\.?\d*)"', r':\1', json_str)
                json_str = re.sub(r'([^"])"([,}])', r'\1\2', json_str)
                json_str = json_str.strip()
                if not json_str.startswith('{'): json_str = '{' + json_str
                if not json_str.endswith('}'): json_str = json_str + '}'
                response_parsed = json.loads(json_str)
            if not all(key in response_parsed for key in expected_keys):
                raise ValueError(f"Missing expected keys in response: {expected_keys}")
            total_prob = sum(response_parsed.values())
            if total_prob > 0:
                normalized_probs = [v / total_prob for v in response_parsed.values()]
                return normalized_probs
            else:
                raise ValueError("Sum of probabilities is zero, cannot normalize.")
        else:
            raise ValueError("No JSON found")
    except (ValueError, SyntaxError, TypeError, json.JSONDecodeError, ZeroDivisionError) as e:
        raise ValueError(f"Failed to parse response: {e}")

def call_model(system_prompt, user_prompt, model_name, prompt_option, temperature=0, **kwargs):
    timeout_seconds = 10
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)
    try:
        if kwargs.get("openrouter") and openrouterclient:
            response = openrouterclient.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=250,
                temperature=temperature,
                timeout=timeout_seconds
            )
            return response.choices[0].message.content
        elif genai and 'gemini' in model_name.lower():
            if prompt_option == 'verbalized':
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt
                )
                response = model.generate_content(
                    [user_prompt],
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=250,
                        temperature=temperature,
                    ),
                    request_options={"timeout": timeout_seconds}
                )
                return response.text.strip()
        elif openai and model_name in ["davinci-002", "babbage-002", "gpt-3.5-turbo-instruct"]:
            if prompt_option == 'token_prob':
                response = openai.Completion.create(
                    engine=model_name,
                    prompt=system_prompt + user_prompt,
                    max_tokens=1,
                    stop=None,
                    temperature=temperature
                )
                token_logprobs = response.choices[0].logprobs.top_logprobs[0]
                log_probs = torch.tensor(list(token_logprobs.values()))
                probs = torch.nn.functional.softmax(log_probs, dim=-1)
                return {token: prob.item() for token, prob in zip(token_logprobs.keys(), probs)}
            if prompt_option == 'verbalized':
                response = openai.ChatCompletion.create(
                    model=model_name,
                    prompt=system_prompt + user_prompt,
                    max_tokens=250,
                    temperature=temperature,
                    response_format={"type": "json_object"})
                return response.choices[0]['message']['content']
        elif openai:
            if prompt_option == 'token_prob':
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1,
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=5
                )
                token_logprobs = {
                    item.token: item.logprob
                    for item in response.choices[0].logprobs.content[0].top_logprobs
                }
                log_probs = torch.tensor(list(token_logprobs.values()))
                probs = torch.nn.functional.softmax(log_probs, dim=-1)
                return {token: prob.item() for token, prob in zip(token_logprobs.keys(), probs)}
            if prompt_option == 'verbalized':
                response = openaiclient.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=250,
                    temperature=temperature,
                    response_format={"type": "json_object"})
                return response.choices[0].message.content
        return None
    except TimeoutError:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)

@retry(wait=wait_random_exponential(min=0.1, max=1), stop=stop_after_attempt(100))
def call_model_with_retry(system_prompt, user_prompt, model, prompt_type, temperature=0, **kwargs):
    result = call_model(system_prompt, user_prompt, model, prompt_type, temperature=temperature, **kwargs)
    if result is None:
        raise Exception("API call failed or timed out")
    return result

def check_chat_model(model_name):
    return any(substring in model_name.lower() for substring in
               ['dpo', 'instruct', 'chat', 'tulu', 'sft', 'it', 'gemini'])

def check_spectrum_model(model_name):
    return any(substring in model_name.lower() for substring in
               ['spectrum'])

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help='file location')
parser.add_argument('--output_file', type=str, help='file location')
parser.add_argument('--model_name', type=str, help='model name')
parser.add_argument('--method', type=str, help='method')
parser.add_argument('--csd3', type=str, help='compute environment')
parser.add_argument('--debug', action='store_true', help='Debugging')
parser.add_argument('--openrouter', action='store_true', help='Use OpenRouter API')
args = parser.parse_args()

model_name = args.model_name
prompt_method = args.method

dataset = pd.read_pickle(args.input_file)
if args.debug:
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset = dataset[:50]

if args.csd3:
    hf_cache_folder = os.environ.get('HF_CACHE_CSD3', '/rds/project/rds-gLFG2ebmCDI/huggingface')
    os.chdir(os.environ.get('CSD3_WORKDIR', '/rds/user/th656/hpc-work/SimBench'))
else:
    # hf_cache_folder = os.environ.get('HF_CACHE_LOCAL', '/mnt/nas_home/huggingface')
    hf_cache_folder = os.environ.get('HF_CACHE_LOCAL', '/gscratch/xlab/tsor13/.cache/huggingface')
    # os.chdir(os.environ.get('LOCAL_WORKDIR', '/mnt/nas_home/th656/code/CSS/SimBench'))
    # get current directory
    os.chdir(os.environ.get('LOCAL_WORKDIR', os.getcwd()))

total_probs = []
model_distribution = []
total_system_prompt = []
total_user_prompt = []

if any(keyword in model_name.lower() for keyword in ['gpt', 'claude', 'gemini', 'command']) or args.openrouter:
    for id in tqdm(range(len(dataset))):
        system_prompt, user_prompt = generate_prompt(dataset.iloc[id], prompt_method)
        total_system_prompt.append(system_prompt)
        total_user_prompt.append(user_prompt)
        if prompt_method == 'token_prob':
            token_logprobs = call_model_with_retry(system_prompt, user_prompt, model_name, prompt_method, openrouter=args.openrouter)
            token_probabilities = {
                token: np.exp(token_logprobs.get(token, float('-inf')))
                for token in dataset.loc[id, 'human_answer'].keys()
            }
            total = sum(token_probabilities.values())
            token_probabilities = [value / total for value in token_probabilities.values()]
            model_distribution.append(token_probabilities)
        if prompt_method == 'verbalized':
            max_retries = 5
            retry_count = 0
            temperature = 0.0001
            expected_keys = list(dataset.loc[id, 'human_answer'].keys())
            while retry_count < max_retries:
                try:
                    response = call_model_with_retry(system_prompt, user_prompt, model_name, prompt_method,
                                                     temperature=temperature, openrouter=args.openrouter)
                    normalized_probs = parse_and_normalize_json_response(response, expected_keys)
                    model_distribution.append(normalized_probs)
                    break
                except ValueError:
                    retry_count += 1
                    temperature = 1
                    if retry_count == max_retries:
                        raise ValueError("Max retries reached, unable to parse response.")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_folder, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',
                                                 cache_dir=hf_cache_folder, load_in_8bit=True,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for id in tqdm(range(len(dataset))):
        system_prompt, user_prompt = generate_prompt(dataset.iloc[id], prompt_method)
        total_system_prompt.append(system_prompt)
        total_user_prompt.append(user_prompt)
        chat_model = check_chat_model(model_name)
        spectrum_model = check_spectrum_model(model_name)
        if prompt_method == 'token_prob':
            target_tokens = list(dataset.loc[id, 'human_answer'].keys())
            if chat_model:
                overall_prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    tokenize=False)
                    # TAYLOR - hmm i feel like we should be adding the generation prompt here too?
            elif spectrum_model:
                overall_prompt = tokenizer.apply_chat_template(
                    [{"role": "description", "content": system_prompt}, {"role": "input", "content": user_prompt}],
                    tokenize=False, add_generation_prompt=True)
                # replace <bos> with ""
                overall_prompt = overall_prompt.replace("<bos>", "")
                # TAYLOR - has a double bos problem for gemma, but could ignore for now and just take the result
            else:
                overall_prompt = system_prompt + "\n" + user_prompt if system_prompt else user_prompt
            probabilities = get_token_probabilities_local_model(model, tokenizer, overall_prompt, target_tokens)
            total_prob = sum(probabilities.values())
            normalized_probs = [v / total_prob for v in probabilities.values()]
            total_probs.append(total_prob)
            model_distribution.append(normalized_probs)
        elif prompt_method == 'verbalized':
            max_retries = 5
            retry_count = 0
            temperature = 0.0001
            expected_keys = list(dataset.loc[id, 'human_answer'].keys())
            while retry_count < max_retries:
                if chat_model:
                    input_ids = tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        add_generation_prompt=True,
                        return_tensors="pt",
                        tokenize=True
                    ).to('cuda')
                else:
                    overall_prompt = system_prompt + "\n" + user_prompt if system_prompt else user_prompt
                    input_ids = tokenizer.encode(overall_prompt, return_tensors='pt').to('cuda')
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=250,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True
                )
                response_ids = output_ids[:, input_ids.shape[-1]:]
                response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
                try:
                    normalized_probs = parse_and_normalize_json_response(response, expected_keys)
                    model_distribution.append(normalized_probs)
                    break
                except ValueError:
                    retry_count += 1
                    temperature = 1
                    if retry_count == max_retries:
                        raise ValueError("Max retries reached, unable to parse response.")

if not total_probs:
    total_probs = [np.nan] * len(model_distribution)
new_rows = []
for idx, row in dataset.iterrows():
    new_row = row.copy()
    new_row['System_Prompt'] = total_system_prompt[idx]
    new_row['User_Prompt'] = total_user_prompt[idx]
    new_row['Response_Distribution'] = model_distribution[idx]
    new_row['Sum_of_Probs'] = total_probs[idx]
    new_row['Model'] = model_name
    new_row['Prompt_Method'] = prompt_method
    new_rows.append(new_row)
new_dataset = pd.DataFrame(new_rows)
new_dataset.to_pickle(args.output_file)