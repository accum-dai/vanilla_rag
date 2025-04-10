from typing import Dict, List, Any, Optional, Tuple
import uuid

# 定义文档的基本结构
class Document:
    def __init__(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        # 如果没有提供ID则生成一个
        self.doc_id = doc_id or str(uuid.uuid4())
    
    def __repr__(self):
        return f"Document(id={self.doc_id}, metadata={self.metadata})"

# 定义用于处理的批次结构
class DocumentBatch:
    def __init__(self, documents: List[Document]):
        self.documents = documents
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return self.documents[idx]