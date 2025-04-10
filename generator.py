import os
import requests
import json
import time
from typing import List, Dict, Any, Optional, Union, Callable

class Generator:
    """基础生成器接口"""
    
    def generate(self, prompt: str, stream: bool = False) -> Union[str, Callable]:
        """
        生成文本，可选流式输出
        
        Args:
            prompt: 提示词
            stream: 是否流式输出
            
        Returns:
            文本或生成器函数
        """
        raise NotImplementedError("子类必须实现此方法")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Generator':
        """
        从配置创建生成器实例
        
        Args:
            config: 配置字典
            
        Returns:
            生成器实例
        """
        raise NotImplementedError("子类必须实现此方法")


class OllamaGenerator(Generator):
    """调用Ollama API的生成器"""
    
    def __init__(
        self, 
        model_name: str = "qwen:32b", 
        api_url: str = "http://localhost:11434/api/generate",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        初始化Ollama生成器
        
        Args:
            model_name: 模型名称
            api_url: API URL
            api_key: API密钥（可选）
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.extra_params = kwargs
    
    def generate(self, prompt: str, stream: bool = False) -> Union[str, Callable]:
        """
        调用Ollama API生成文本，支持流式输出
        
        Args:
            prompt: 提示词
            stream: 是否流式输出
            
        Returns:
            文本或生成器函数
        """
        # 准备请求数据
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            **self.extra_params
        }
        
        # 准备请求头
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        if not stream:
            # 非流式输出
            try:
                response = requests.post(
                    self.api_url,
                    json=request_data,
                    headers=headers
                )
                
                response.raise_for_status()
                return response.json().get('response', 'Error: No response received')
            except requests.RequestException as e:
                return f"Error: {str(e)}"
        else:
            # 流式输出，返回一个生成器函数
            def stream_generator():
                try:
                    response = requests.post(
                        self.api_url,
                        json=request_data,
                        headers=headers,
                        stream=True
                    )
                    
                    response.raise_for_status()
                    
                    # 处理流式响应
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'response' in chunk:
                                    yield chunk['response']
                                # 检查是否是最后一个消息
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                yield "Error: Could not decode response chunk"
                except requests.RequestException as e:
                    yield f"Error: {str(e)}"
            
            return stream_generator
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OllamaGenerator':
        """
        从配置创建Ollama生成器实例
        
        Args:
            config: 配置字典
            
        Returns:
            OllamaGenerator实例
        """
        # 提取主要参数
        model_name = config.get("default_model", "qwen:32b")
        api_url = config.get("api_url", "http://localhost:11434/api/generate")
        
        # API密钥可以从配置或环境变量中获取
        api_key = config.get("api_key")
        if api_key is None:
            api_key = os.environ.get("OLLAMA_API_KEY")
        
        # 提取其他参数
        extra_params = {k: v for k, v in config.items() 
                        if k not in ["default_model", "api_url", "api_key"]}
        
        return cls(
            model_name=model_name,
            api_url=api_url,
            api_key=api_key,
            **extra_params
        )


class OpenAIGenerator(Generator):
    """调用OpenAI API的生成器"""
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo",
        api_url: str = "https://api.openai.com/v1/chat/completions",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        初始化OpenAI生成器
        
        Args:
            model_name: 模型名称
            api_url: API URL
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大生成长度
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
    
    def generate(self, prompt: str, stream: bool = False) -> Union[str, Callable]:
        """
        调用OpenAI API生成文本，支持流式输出
        
        Args:
            prompt: 提示词
            stream: 是否流式输出
            
        Returns:
            文本或生成器函数
        """
        if not self.api_key:
            return "Error: OpenAI API key is required"
        
        # 准备请求数据
        request_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            **self.extra_params
        }
        
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if not stream:
            # 非流式输出
            try:
                response = requests.post(
                    self.api_url,
                    json=request_data,
                    headers=headers
                )
                
                response.raise_for_status()
                response_json = response.json()
                
                # 提取文本
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    return response_json['choices'][0]['message']['content']
                return "Error: No valid response received"
            except requests.RequestException as e:
                return f"Error: {str(e)}"
        else:
            # 流式输出，返回一个生成器函数
            def stream_generator():
                try:
                    response = requests.post(
                        self.api_url,
                        json=request_data,
                        headers=headers,
                        stream=True
                    )
                    
                    response.raise_for_status()
                    
                    # 处理流式响应
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                line = line[6:]  # 去除 "data: " 前缀
                                
                                if line == "[DONE]":
                                    break
                                
                                try:
                                    chunk = json.loads(line)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            yield delta['content']
                                except json.JSONDecodeError:
                                    continue
                except requests.RequestException as e:
                    yield f"Error: {str(e)}"
            
            return stream_generator
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpenAIGenerator':
        """
        从配置创建OpenAI生成器实例
        
        Args:
            config: 配置字典
            
        Returns:
            OpenAIGenerator实例
        """
        # 提取主要参数
        model_name = config.get("default_model", "gpt-3.5-turbo")
        api_url = config.get("api_url", "https://api.openai.com/v1/chat/completions")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1000)
        
        # API密钥可以从配置或环境变量中获取
        api_key = config.get("api_key")
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        # 提取其他参数
        extra_params = {k: v for k, v in config.items() 
                       if k not in ["default_model", "api_url", "api_key", 
                                   "temperature", "max_tokens"]}
        
        return cls(
            model_name=model_name,
            api_url=api_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params
        )


def create_generator(config_manager, generator_type: Optional[str] = None) -> Generator:
    """
    根据配置创建生成器实例
    
    Args:
        config_manager: 配置管理器
        generator_type: 生成器类型，如果为None则使用默认类型
        
    Returns:
        生成器实例
    """
    if generator_type is None:
        generator_type = config_manager.get("generator", "default_type", default="ollama")
    
    generator_config = config_manager.get_generator_config(generator_type)
    
    if generator_type == "ollama":
        return OllamaGenerator.from_config(generator_config)
    elif generator_type == "openai":
        return OpenAIGenerator.from_config(generator_config)
    else:
        raise ValueError(f"不支持的生成器类型: {generator_type}")