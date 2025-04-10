import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from data_types import Document, DocumentBatch


class EmbeddingModel:
    """基础嵌入模型接口"""
    
    def embed_query(self, text: str) -> List[float]:
        """
        将查询文本嵌入为向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def embed_documents(self, documents: DocumentBatch) -> List[List[float]]:
        """
        将多个文档嵌入为向量
        
        Args:
            documents: 文档批次
            
        Returns:
            嵌入向量列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        将文本批次嵌入为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'EmbeddingModel':
        """
        从配置创建嵌入模型实例
        
        Args:
            config: 配置字典
            
        Returns:
            嵌入模型实例
        """
        raise NotImplementedError("子类必须实现此方法")


class SentenceTransformerEmbedding(EmbeddingModel):
    """使用SentenceTransformer的嵌入模型"""
    
    def __init__(
        self, 
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        初始化句子变换器嵌入模型
        
        Args:
            model_name: 模型名称
            device: 设备 ('cpu', 'cuda:0', etc.)，None表示自动检测
            batch_size: 批处理大小
        """
        from sentence_transformers import SentenceTransformer
        
        # 自动检测设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
                print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("未检测到GPU，使用CPU")
        
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 初始化模型
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        
        print(f"初始化嵌入模型: {model_name}")
        print(f"使用设备: {device}, 批大小: {batch_size}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本嵌入为向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        return self.model.encode(text, device=self.device).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        将一批文本嵌入为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        return self.model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=True, 
            device=self.device
        ).tolist()
    
    def embed_documents(self, documents: DocumentBatch) -> List[List[float]]:
        """
        将文档批次嵌入为向量
        
        Args:
            documents: 文档批次
            
        Returns:
            嵌入向量列表
        """
        texts = [doc.content for doc in documents]
        return self.embed_batch(texts)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SentenceTransformerEmbedding':
        """
        从配置创建SentenceTransformer嵌入模型实例
        
        Args:
            config: 配置字典
            
        Returns:
            SentenceTransformerEmbedding实例
        """
        model_name = config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
        device = config.get("device")
        batch_size = config.get("batch_size", 32)
        
        return cls(
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )


class OpenAIEmbedding(EmbeddingModel):
    """使用OpenAI API的嵌入模型"""
    
    def __init__(
        self, 
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        batch_size: int = 20
    ):
        """
        初始化OpenAI嵌入模型
        
        Args:
            model_name: 模型名称
            api_key: OpenAI API密钥
            batch_size: 批处理大小
        """
        import openai
        
        # 设置API密钥
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"初始化OpenAI嵌入模型: {model_name}")
        print(f"批大小: {batch_size}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本嵌入为向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        将一批文本嵌入为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        # 批量处理
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"已处理 {i+len(batch)}/{len(texts)} 个嵌入")
            except Exception as e:
                print(f"嵌入处理错误: {e}")
                # 返回零向量作为错误处理
                for _ in range(len(batch)):
                    all_embeddings.append([0.0] * 1536)  # OpenAI embeddings 维度
        
        return all_embeddings
    
    def embed_documents(self, documents: DocumentBatch) -> List[List[float]]:
        """
        将文档批次嵌入为向量
        
        Args:
            documents: 文档批次
            
        Returns:
            嵌入向量列表
        """
        texts = [doc.content for doc in documents]
        return self.embed_batch(texts)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpenAIEmbedding':
        """
        从配置创建OpenAI嵌入模型实例
        
        Args:
            config: 配置字典
            
        Returns:
            OpenAIEmbedding实例
        """
        model_name = config.get("model_name", "text-embedding-ada-002")
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        batch_size = config.get("batch_size", 20)
        
        return cls(
            model_name=model_name,
            api_key=api_key,
            batch_size=batch_size
        )


def create_embedding_model(config_manager, provider: Optional[str] = None) -> EmbeddingModel:
    """
    根据配置创建嵌入模型实例
    
    Args:
        config_manager: 配置管理器
        provider: 提供商，如果为None则使用默认提供商
        
    Returns:
        嵌入模型实例
    """
    if provider is None:
        provider = config_manager.get("embedding", "default_provider", default="sentence_transformer")
    
    embedding_config = config_manager.get_embedding_config(provider)
    
    if provider == "sentence_transformer":
        return SentenceTransformerEmbedding.from_config(embedding_config)
    elif provider == "openai":
        return OpenAIEmbedding.from_config(embedding_config)
    else:
        raise ValueError(f"不支持的嵌入模型提供商: {provider}")