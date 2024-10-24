from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config.prompt_list import causality_system_prompt, causality_q1

model_path = "../Models/google/gemma-2-27b-it"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

input_text = "Write me a poem about Machine Learning."
messages = [
    {"role": "system", "content": causality_system_prompt},
    {"role": "user", "content": "由于阿根廷空军研究与发展部成功研制了Aucán无人机，阿根廷空军于10月24日在拉里奥哈省Chanmical空军基地举行了该无人机的测试。Class I型无人机重150kg，主要用于研究与训练，而更大的Class II型则因安装了一台60马力的发动机并可携带50kg任务载荷，导致其总重达到300kg，能以最大速度210km/h执行高达4500m的ISR任务，且续航时间长达11小时。" + f"\n({causality_q1})"}
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
