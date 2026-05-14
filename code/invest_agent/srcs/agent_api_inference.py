from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-0WnJfsGYookxeCk1ZMzQXg",
    base_url="https://llmapi.paratera.com/v1/"
)

def agent_api_inference(model_id, content):
    # 构建用户消息
    messages = [
        {"role": "user", "content": content}
    ]
    
    # 调用API
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=512,
        temperature=0.6,
        top_p=0.95
    )
    
    # 提取并返回响应内容
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # 测试
    prompt = "What is the capital of France?"
    result = agent_api_inference(model_id="Qwen3.5-Plus", content=prompt)
    print(f"Assistant: {result}")
