# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------



import json
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 定义路径和超参数 
model_path = "output/qwen3_lora_sft/"
quant_path = "output/qwen3_lora_sft_int4/"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

data = []
fd = open("../data/summary_data/train.json")
raw = json.load(fd)
for msg in raw:
    try:
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg["instruction"]},
            {"role": "assistant", "content": msg["output"]}
        ]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        data.append(text.strip())
    except:
        pass

print("校准样本数：", len(data))
print(data[0])

model.quantize(tokenizer, quant_config=quant_config, calib_data=data)


model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)
