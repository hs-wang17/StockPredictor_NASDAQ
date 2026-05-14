import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

MODEL_PATH = "/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/llm_model/Qwen-7B/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# 2. 全局加载模型和分词器（只加载一次）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config
)

def agent_inference(model_name, content):
    # 3. 使用聊天模板格式化输入
    messages = [
        {"role": "user", "content": content}
    ]
    # apply_chat_template 会自动加上模型需要的特殊标记
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 4. 推理
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6, # R1 系列建议稍微调低温度以保持逻辑稳定
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 只取生成的回复部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.replace('Ġ', ' ').replace('Ċ', '\n').split("</think>")[-1].strip()

    return response

if __name__ == "__main__":
    # 测试
    prompt = "What is the capital of France?"
    result = agent_inference(model_name="qwen", content=prompt)
    print(f"Assistant: {result}")