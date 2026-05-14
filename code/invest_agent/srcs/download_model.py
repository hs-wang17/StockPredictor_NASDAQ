import os
from modelscope import snapshot_download

# 请在这里选择你需要的模型ID：
# 1.5B 模型: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
# 7B 模型:   'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# 8B 模型:   'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
print(f"开始下载模型: {model_id}")
os.makedirs('/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/llm_model/Qwen-7B', exist_ok=True)
model_dir = snapshot_download(model_id, cache_dir='/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/llm_model/Qwen-7B')
print(f"模型下载完成，保存在: {model_dir}")