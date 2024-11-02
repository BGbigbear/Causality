import time

import requests
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

url = "https://api.deepinfra.com/v1/openai"
key = "4GT75Atw94Q4j044iBAT1AK85NreqXJU"
client = OpenAI(
    api_key=key,
    base_url=url,
)


def chat_completion(messages, model="google/gemma-2-27b-it", retries=5, backoff=2):
    # model = "Qwen/Qwen2.5-72B-Instruct"
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


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return tokenizer, model


def model_generation(tokenizer, model, messages):
    # input_text = "Write me a poem about Machine Learning."
    # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=256)
    return tokenizer.decode(outputs[0])


if __name__ == '__main__':
    msgs = [
        {"role": "user", "content": "Hello"}
    ]
    # tok, mod = load_model("gemma2")
    # print(model_generation(tok, mod, msgs))
    # print(chat_completion(msgs))

    print(request_api(msgs))
