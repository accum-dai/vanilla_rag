import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import datetime
from data_types import Document, DocumentBatch

class VectorStore:
    def add_documents(self, documents: DocumentBatch, embeddings: List[List[float]] = None) -> List[str]:
        """添加文档到向量存储"""
        raise NotImplementedError
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter: Dict = None) -> List[Document]:
        """搜索相似文档"""
        raise NotImplementedError
    
    def update_document(self, doc_id: str, document: Document, embedding: List[float] = None) -> None:
        """更新文档"""
        raise NotImplementedError
    
    def delete(self, doc_ids: List[str]) -> None:
        """删除文档"""
        raise NotImplementedError
    
    def get(self, doc_ids: List[str] = None, where: Dict = None) -> DocumentBatch:
        """获取文档"""
        raise NotImplementedError

class ChromaVectorStore(VectorStore):
    def __init__(
        self, 
        collection_name: str, 
        embedding_model: Optional[Any] = None,
        persist_dir: str = "./chroma_db"
    ):
        self.embedding_model = embedding_model
        # self.client = chromadb.Client(Settings(persist_directory=persist_dir))
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"连接到现有集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"创建新集合: {collection_name}")
    
    def add_documents(self, documents: DocumentBatch, embeddings: List[List[float]] = None) -> List[str]:
        """添加文档到向量存储"""
        if not documents:
            return []
        
        ids = [doc.doc_id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # 如果没有提供embeddings且有embedding_model，则计算embeddings
        if embeddings is None and self.embedding_model:
            embeddings = self.embedding_model.embed_documents(documents)
        
        # 添加到集合
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        return ids
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter: Dict = None) -> List[Document]:
        """搜索相似文档"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        documents = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                
                # 添加相似度分数到元数据
                if 'distances' in results and results['distances'][0]:
                    metadata['similarity_score'] = results['distances'][0][i]
                
                doc = Document(content=content, metadata=metadata, doc_id=doc_id)
                documents.append(doc)
        
        return documents
    
    def update_document(self, doc_id: str, document: Document, embedding: List[float] = None) -> None:
        """更新文档"""
        # 如果没有提供embedding且有embedding_model，则计算embedding
        if embedding is None and self.embedding_model:
            embedding = self.embedding_model.embed_query(document.content)
        
        # 更新文档
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding] if embedding else None,
            documents=[document.content],
            metadatas=[document.metadata]
        )
    
    def delete(self, doc_ids: List[str]) -> None:
        """删除文档"""
        self.collection.delete(ids=doc_ids)
    
    def get(self, doc_ids: List[str] = None, where: Dict = None) -> DocumentBatch:
        """获取文档"""
        results = self.collection.get(ids=doc_ids, where=where)
        
        documents = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids']):
                content = results['documents'][i]
                metadata = results['metadatas'][i] if 'metadatas' in results else {}
                
                doc = Document(content=content, metadata=metadata, doc_id=doc_id)
                documents.append(doc)
        
        return DocumentBatch(documents)
    
    def persist(self) -> None:
        """持久化存储"""
        # ChromaDB在不同版本中持久化方式不同
        # 较新版本可能在配置persist_directory时自动持久化
        
        try:
            # 尝试旧版本的持久化方法
            self.client.persist()
            print("ChromaDB数据已通过client.persist()方法保存")
        except AttributeError:
            # 新版本可能不需要显式持久化
            print(f"ChromaDB数据已自动保存到配置的持久化目录")