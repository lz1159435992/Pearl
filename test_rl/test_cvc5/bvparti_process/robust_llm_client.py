
import time
import random
import requests
from typing import Optional, Dict, Any

class RobustOllamaClient:
    """健壮的Ollama客户端，具有重试和错误处理功能"""
    
    def __init__(self, host: str, model: str, max_retries: int = 3):
        self.host = host
        self.model = model
        self.max_retries = max_retries
        self.base_delay = 1.0
        
    def generate_with_retry(self, prompt: str, **kwargs) -> Optional[str]:
        """带重试的生成方法"""
        for attempt in range(self.max_retries):
            try:
                import ollama
                client = ollama.Client(host=self.host)
                
                response = client.generate(
                    model=self.model,
                    prompt=prompt,
                    options=kwargs.get('options', {})
                )
                
                if response and 'response' in response:
                    return response['response']
                    
            except Exception as e:
                print(f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # 指数退避
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print("所有重试都失败，返回默认响应")
                    return "Error: LLM service unavailable"
        
        return None
    
    def stream_with_retry(self, prompt: str, **kwargs):
        """带重试的流式生成方法"""
        for attempt in range(self.max_retries):
            try:
                import ollama
                client = ollama.Client(host=self.host)
                
                for chunk in client.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=True,
                    options=kwargs.get('options', {})
                ):
                    if chunk and 'response' in chunk:
                        yield chunk['response']
                
                return  # 成功完成
                
            except Exception as e:
                print(f"LLM流式调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print("所有重试都失败")
                    yield "Error: LLM service unavailable"
