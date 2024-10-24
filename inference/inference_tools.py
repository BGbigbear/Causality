import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

client = OpenAI(
    api_key="4GT75Atw94Q4j044iBAT1AK85NreqXJU",
    base_url="https://api.deepinfra.com/v1/openai",
)


def chat_completion(messages, model="google/gemma-2-27b-it"):
    # model = "Qwen/Qwen2.5-72B-Instruct"
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion


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
