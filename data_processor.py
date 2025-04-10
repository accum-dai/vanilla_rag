import pandas as pd
import os
import datetime
import glob
from typing import List, Dict, Any, Callable, Optional, Iterator, Generator, Union
from data_types import Document, DocumentBatch

# 数据加载器
class DataLoader:
    def load(self) -> Union[DocumentBatch, Generator[DocumentBatch, None, None]]:
        """
        加载数据并返回文档批次或批次生成器
        """
        raise NotImplementedError

class WikiCSVDataLoader(DataLoader):
    def __init__(
        self, 
        file_path: str, 
        id_prefix: str = "",
        chunk_size: int = 1,
        batch_mode: bool = True
    ):
        """
        初始化维基百科CSV数据加载器
        
        Args:
            file_path: CSV文件路径或目录路径
            id_prefix: ID前缀，用于区分不同来源的文档
            chunk_size: 处理CSV时的块大小
            batch_mode: 是否使用批处理模式（生成器）
        """
        self.file_path = file_path
        self.id_prefix = id_prefix
        self.chunk_size = chunk_size
        self.batch_mode = batch_mode
    
    def _process_csv_row(self, idx: int, row: pd.Series) -> Document:
        """
        处理CSV的单行数据，转换为Document对象
        
        Args:
            idx: 行索引
            row: 行数据
            
        Returns:
            Document对象
        """
        # 提取ID，如果没有则生成一个
        doc_id = None
        if 'id' in row and pd.notna(row['id']):
            doc_id = f"{self.id_prefix}{row['id']}"
        else:
            doc_id = f"{self.id_prefix}doc_{idx}"
        
        # 提取标题和正文
        title = str(row.get('title', '')) if pd.notna(row.get('title', '')) else ''
        content = str(row.get('text', '')) if pd.notna(row.get('text', '')) else ''
        
        # 组合标题和内容
        full_content = f"Title: {title}\nContent: {content}"
        
        # 构建元数据
        metadata = {
            "source_file": os.path.basename(self.file_path),
            "import_time": datetime.datetime.now().isoformat()
        }
        
        # 添加URL（如果存在）
        if 'url' in row and pd.notna(row['url']):
            metadata['url'] = row['url']
        
        # 添加标题（如果存在）
        if title:
            metadata['title'] = title
        
        # 创建文档
        return Document(content=full_content, metadata=metadata, doc_id=doc_id)
    
    def _get_csv_files(self) -> List[str]:
        """
        获取要处理的CSV文件列表
        
        Returns:
            CSV文件路径列表
        """
        if os.path.isfile(self.file_path):
            return [self.file_path]
        elif os.path.isdir(self.file_path):
            # 查找目录中的所有CSV文件
            return glob.glob(os.path.join(self.file_path, "*.csv"))
        else:
            print(f"警告: 路径不存在或不是有效的文件/目录: {self.file_path}")
            return []
    
    def _load_csv_in_chunks(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """
        分块加载CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            DataFrame的生成器
        """
        try:
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                yield chunk
        except Exception as e:
            print(f"读取CSV文件出错: {file_path}, 错误: {e}")
            yield pd.DataFrame()  # 返回空DataFrame
    
    def load(self) -> Union[DocumentBatch, Generator[DocumentBatch, None, None]]:
        """
        加载数据并返回文档批次或批次生成器
        
        Returns:
            文档批次或批次生成器
        """
        csv_files = self._get_csv_files()
        
        if not csv_files:
            print(f"没有找到CSV文件: {self.file_path}")
            return DocumentBatch([])
        
        if not self.batch_mode:
            # 非批处理模式，加载所有文档
            all_documents = []
            
            for csv_file in csv_files:
                print(f"处理文件: {csv_file}...")
                
                for i, chunk in enumerate(self._load_csv_in_chunks(csv_file)):
                    start_idx = i * self.chunk_size
                    chunk_docs = [self._process_csv_row(start_idx + idx, row) 
                                 for idx, row in chunk.iterrows()]
                    all_documents.extend(chunk_docs)
                    print(f"  已处理 {len(all_documents)} 个文档...")
            
            return DocumentBatch(all_documents)
        else:
            # 批处理模式，返回文档批次生成器
            def batch_generator() -> Generator[DocumentBatch, None, None]:
                for csv_file in csv_files:
                    print(f"处理文件: {csv_file}...")
                    
                    for i, chunk in enumerate(self._load_csv_in_chunks(csv_file)):
                        start_idx = i * self.chunk_size
                        chunk_docs = [self._process_csv_row(start_idx + idx, row) 
                                     for idx, row in chunk.iterrows()]
                        yield DocumentBatch(chunk_docs)
                        print(f"  已处理 {(i+1) * len(chunk)} 个文档...")
            
            return batch_generator()

# 文本分割器
class TextSplitter:
    def split(self, documents: DocumentBatch) -> DocumentBatch:
        """将文档拆分为块"""
        raise NotImplementedError

class SimpleTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_text(self, text: str) -> List[str]:
        """将单个文本拆分为多个块"""
        chunks = []
        start = 0
        is_break = False
        
        while start < len(text):
            end = start + self.chunk_size
            if end > len(text):
                end = len(text)
                is_break = True
            
            chunks.append(text[start:end])
            start = end - self.overlap
            
            if is_break:
                break
        
        return chunks
    
    def split(self, documents: DocumentBatch) -> DocumentBatch:
        result_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.content)
            
            for i, chunk in enumerate(chunks):
                # 创建新的元数据，保留原始信息
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)
                metadata["parent_id"] = doc.doc_id
                
                # 创建新ID，包含父ID和分块信息
                chunk_id = f"{doc.doc_id}_chunk_{i}"
                
                # 创建新文档
                chunk_doc = Document(
                    content=chunk,
                    metadata=metadata,
                    doc_id=chunk_id
                )
                
                result_docs.append(chunk_doc)
        
        return DocumentBatch(result_docs)