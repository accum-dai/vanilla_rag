from typing import List, Dict, Any, Optional, Union, Callable
from data_types import Document
from generator import Generator, OllamaGenerator, OpenAIGenerator


class RAG:
    def __init__(
        self, 
        vector_store,
        embedding_model, 
        generator: Generator,
        top_k: int = 5,
        prompt_template: Optional[str] = None
    ):
        """
        初始化RAG系统
        
        Args:
            vector_store: 向量存储
            embedding_model: 嵌入模型
            generator: 文本生成器
            top_k: 检索的文档数量
            prompt_template: 提示词模板
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.generator = generator
        self.top_k = top_k
        self.prompt_template = prompt_template or """
你是一个知识丰富的AI助手。请使用以下检索到的相关信息来回答用户问题。如果检索到的信息不足以回答问题，请说明你不知道。

相关信息:
{context}

用户问题: {query}

请提供准确、全面的回答。如果你使用了包含URL的文档，请在回答中引用这些来源，格式为[文档标题](URL)。
"""
    
    def retrieve(self, query: str, top_k: Optional[int] = None, filter: Optional[Dict] = None) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 检索的文档数量
            filter: 过滤条件
            
        Returns:
            相关文档列表
        """
        if top_k is None:
            top_k = self.top_k
            
        # 对查询进行嵌入
        query_embedding = self.embedding_model.embed_query(query)
        
        # 检索相关文档
        return self.vector_store.search(query_embedding, top_k=top_k, filter=filter)
    
    def format_context_with_citations(self, docs: List[Document]) -> str:
        """
        格式化上下文，添加引用编号
        
        Args:
            docs: 文档列表
            
        Returns:
            格式化后的上下文
        """
        context_parts = []
        
        for i, doc in enumerate(docs):
            doc_content = doc.content
            doc_id = f"[{i+1}]"
            
            # 添加引用编号
            context_part = f"{doc_id} {doc_content}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def generate(self, query: str, context_docs: List[Document], stream: bool = False) -> Union[str, Callable]:
        """
        生成回答
        
        Args:
            query: 查询文本
            context_docs: 上下文文档
            stream: 是否流式输出
            
        Returns:
            回答文本或生成器函数
        """
        # 提取文档内容并添加引用编号
        context = self.format_context_with_citations(context_docs)
        
        # 生成引用信息
        citations = []
        for i, doc in enumerate(context_docs):
            title = doc.metadata.get('title', 'Untitled Document')
            url = doc.metadata.get('url', None)
            
            citation = f"[{i+1}] {title}"
            if url:
                citation += f": {url}"
            
            citations.append(citation)
        
        citations_text = "\n".join(citations) if citations else ""
        
        # 生成提示词
        prompt = self.prompt_template.format(context=context, query=query)
        
        if citations_text:
            prompt += f"\n\n引用信息:\n{citations_text}"
        
        # 生成回答
        return self.generator.generate(prompt, stream=stream)
    
    def query(self, query: str, top_k: Optional[int] = None, filter: Optional[Dict] = None, stream: bool = False) -> Dict[str, Any]:
        """
        端到端查询
        
        Args:
            query: 查询文本
            top_k: 检索的文档数量
            filter: 过滤条件
            stream: 是否流式输出
            
        Returns:
            查询结果
        """
        # 检索相关文档
        retrieved_docs = self.retrieve(query, top_k=top_k, filter=filter)
        
        # 如果没有检索到文档，返回提示
        if not retrieved_docs:
            return {
                "query": query,
                "response": "没有找到相关信息，无法回答您的问题。",
                "documents": []
            }
        
        # 生成回答
        response = self.generate(query, retrieved_docs, stream=stream)
        
        # 返回结果
        result = {
            "query": query,
            "response": response,
            "documents": retrieved_docs,
            "stream": stream
        }
        
        # 添加引用信息
        citations = []
        for i, doc in enumerate(retrieved_docs):
            title = doc.metadata.get('title', 'Untitled Document')
            url = doc.metadata.get('url', None)
            
            if url:
                citations.append({"index": i+1, "title": title, "url": url})
            else:
                citations.append({"index": i+1, "title": title})
        
        result["citations"] = citations
        
        return result