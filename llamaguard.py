from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

g_model_id = "meta-llama/Meta-Llama-Guard-2-8B"


g_tokenizer = AutoTokenizer.from_pretrained(g_model_id)
g_model = AutoModelForCausalLM.from_pretrained(g_model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
    input_ids = g_tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = g_model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return g_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

ans=moderate([
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
])
# `safe`
print(ans)