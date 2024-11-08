import time
import requests
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

url = "https://api.deepinfra.com/v1/openai"
# url = "http://127.0.0.1:8001/v1"
key = ""
model_path = "/workspace/Models/qwen/Qwen2.5-7B-Instruct/"
adapter_path = ""
client = OpenAI(
    api_key=key,
    base_url=url,
)


def chat_completion(messages, model="google/gemma-2-27b-it", retries=5, backoff=2):
    model = "Qwen/Qwen2.5-72B-Instruct"
    attempt = 0
    while attempt < retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0  # defaults to 1
            )
            return str(completion.choices[0].message.content)
        except Exception as e:
            print(f"Connection error: {e}. Retrying in {backoff ** attempt} seconds...")
            time.sleep(backoff ** attempt)
            attempt += 1
    print("Max retries reached. Exiting.")
    return None


def request_api(messages, model="google/gemma-2-27b-it", retries=5, backoff=2):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    data = {
        "model": model,
        "messages": messages
    }

    attempt = 0
    while attempt < retries:
        try:
            response = requests.post(url + "/chat/completions", headers=headers, json=data)
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            if len(response.json()['choices']) > 1:
                return [choice['message']['content'] for choice in response.json()['choices']]
            else:
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Request error: {e}. Retrying in {backoff ** attempt} seconds...")
            time.sleep(backoff ** attempt)
            attempt += 1
    print("Max retries reached. Exiting.")
    return None


def load_model(vllm=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = None
    if not vllm:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        from vllm import LLM  # local import

        model = LLM(
            model=model_path, max_model_len=10240,
            tensor_parallel_size=2,
            # enable_lora=True
        )

    return tokenizer, model


def model_generation(tokenizer, model, messages, n=10, using_vllm=False):
    # input_text = "Write me a poem about Machine Learning."
    # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    if not using_vllm:
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=256)
        return tokenizer.decode(outputs[0])
    else:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling_params = SamplingParams(n=n, temperature=1.0, top_p=0.8, repetition_penalty=1.05, max_tokens=131072)
        input_ids = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                     for message in messages]
        response = model.generate(
            input_ids, sampling_params,
            # lora_request=LoRARequest('adapter', 1, adapter_path)
        )
        if n == 1:
            return [f"{output.outputs[0].text!r}" for output in response]
        else:
            from util.rouge import select_best_output

            return [select_best_output(output, n, 3) for output in response]


if __name__ == '__main__':
    msgs = [
        {"role": "user", "content": "Hello"}
    ]
    tok, mod = load_model(True)
    print(model_generation(tok, mod, msgs, True))
    # print(chat_completion(msgs))

    print(request_api(msgs))
