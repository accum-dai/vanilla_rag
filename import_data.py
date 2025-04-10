#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG系统数据导入工具

这个脚本用于将数据源配置文件中定义的数据导入到向量存储中，支持批量处理大型文件以避免内存溢出问题。
"""

import argparse
import datetime
import sys
import time
import os
import yaml
import json
from typing import List, Dict, Any, Optional, Tuple, Generator, Union

from data_types import Document, DocumentBatch
from data_processor import WikiCSVDataLoader, SimpleTextSplitter
from embeddings import create_embedding_model
from vector_store import ChromaVectorStore
from data_source_config_manager import DataSourceConfigManager
from app_config_manager import AppConfigManager


def import_data_source(
    config_manager: DataSourceConfigManager,
    data_source: Dict[str, Any],
    vector_store: ChromaVectorStore,
    embedding_model,
    text_splitter: SimpleTextSplitter,
    batch_size: int = 100
) -> int:
    """
    导入单个数据源到向量存储
    
    Args:
        config_manager: 配置管理器
        data_source: 数据源配置
        vector_store: 向量存储
        embedding_model: 嵌入模型
        text_splitter: 文本分割器
        batch_size: 向量存储批处理大小
        
    Returns:
        处理的文档数量
    """
    source_path = data_source["source_path"]
    loader_type = data_source["loader_type"]
    id_prefix = data_source["id_prefix"]
    
    print(f"处理数据源: {source_path} (类型: {loader_type})...")
    
    # 创建数据加载器
    if loader_type == "WikiCSVDataLoader":
        loader = WikiCSVDataLoader(
            file_path=source_path,
            id_prefix=id_prefix,
            batch_mode=True  # 使用批处理模式
        )
    else:
        print(f"不支持的数据加载器类型: {loader_type}")
        return 0
    
    # 记录处理的文档总数
    total_docs = 0
    
    # 加载文档
    loader_result = loader.load()
    
    # 检查结果类型
    if isinstance(loader_result, DocumentBatch):
        # 非批处理模式，得到了单个DocumentBatch
        documents = loader_result
        print(f"  加载了 {len(documents)} 个文档")
        
        # 分割文档
        chunked_documents = text_splitter.split(documents)
        print(f"  分割为 {len(chunked_documents)} 个文本块")
        
        # 批量处理
        for j in range(0, len(chunked_documents), batch_size):
            end_idx = min(j + batch_size, len(chunked_documents))
            batch = DocumentBatch(chunked_documents[j:end_idx])
            
            # 生成嵌入并添加到向量存储
            embeddings = embedding_model.embed_documents(batch)
            vector_store.add_documents(batch, embeddings)
            
            print(f"  添加了 {j+1}-{end_idx}/{len(chunked_documents)} 个文本块")
            
            # 周期性持久化
            if j % 1000 == 0:
                vector_store.persist()
                print("  已持久化向量存储")
        
        total_docs += len(chunked_documents)
        
    else:
        # 批处理模式，得到了一个生成器
        batch_count = 0
        for document_batch in loader_result:
            batch_count += 1
            print(f"  处理批次 {batch_count}，包含 {len(document_batch)} 个文档")
            
            # 分割文档
            chunked_batch = text_splitter.split(document_batch)
            print(f"  分割为 {len(chunked_batch)} 个文本块")
            
            # 批量处理
            for j in range(0, len(chunked_batch), batch_size):
                end_idx = min(j + batch_size, len(chunked_batch))
                sub_batch = DocumentBatch(chunked_batch[j:end_idx])
                
                # 生成嵌入并添加到向量存储
                embeddings = embedding_model.embed_documents(sub_batch)
                vector_store.add_documents(sub_batch, embeddings)
                
                print(f"  添加了 {j+1}-{end_idx}/{len(chunked_batch)} 个文本块")
            
            total_docs += len(chunked_batch)
            
            # 每处理一个批次就持久化一次
            vector_store.persist()
            print("  已持久化向量存储")
    
    # 更新数据源状态
    config_manager.update_data_source_status(
        source_path=source_path,
        status="processed",
        metadata={
            "processed_at": datetime.datetime.now().isoformat(),
            "document_count": total_docs
        }
    )
    
    # 保存配置
    config_manager.save_config(backup=True)
    
    return total_docs


def import_data(
    app_config: AppConfigManager,
    data_config_path: str,
    data_sources_dir: str = "data_sources"
) -> Tuple[ChromaVectorStore, Any]:
    """
    导入数据到向量存储
    
    Args:
        app_config: 应用配置管理器
        data_config_path: 数据配置文件路径
        data_sources_dir: 数据源配置文件存放目录
        
    Returns:
        向量存储和嵌入模型的元组
    """
    # 获取配置
    collection_name = app_config.get("vector_store", "default_collection")
    persist_dir = app_config.get("vector_store", "persist_dir")
    
    # 文本分割器配置
    chunk_size = app_config.get("text_splitter", "chunk_size", default=500)
    overlap = app_config.get("text_splitter", "overlap", default=50)
    
    # 确保值不为None (修复bug)
    if chunk_size is None:
        chunk_size = 500
        print("警告: 配置中未找到text_splitter.chunk_size，使用默认值500")
    
    if overlap is None:
        overlap = 50
        print("警告: 配置中未找到text_splitter.overlap，使用默认值50")
    
    print(f"使用文本分割器参数: chunk_size={chunk_size}, overlap={overlap}")
    
    # 初始化嵌入模型
    embedding_model = create_embedding_model(app_config)
    
    # 初始化向量存储
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        embedding_model=embedding_model,
        persist_dir=persist_dir
    )
    
    # 初始化文本分割器
    text_splitter = SimpleTextSplitter(chunk_size=chunk_size, overlap=overlap)
    
    # 初始化配置管理器
    config_manager = DataSourceConfigManager(data_config_path, data_sources_dir=data_sources_dir)
    
    # 获取未处理的数据源
    data_sources = config_manager.get_unprocessed_data_sources()
    
    if not data_sources:
        print("没有找到需要处理的数据源")
        return vector_store, embedding_model
    
    print(f"找到 {len(data_sources)} 个未处理的数据源")
    
    total_docs = 0
    
    # 处理每个数据源
    for data_source in data_sources:
        docs_processed = import_data_source(
            config_manager=config_manager,
            data_source=data_source,
            vector_store=vector_store,
            embedding_model=embedding_model,
            text_splitter=text_splitter
        )
        
        total_docs += docs_processed
    
    # 最终持久化向量存储
    vector_store.persist()
    print(f"总共导入了 {total_docs} 个文本块")
    
    return vector_store, embedding_model


def main():
    parser = argparse.ArgumentParser(description='RAG数据导入工具')
    # 数据源配置
    parser.add_argument('--data-config', default='data_sources.yaml', help='数据源配置文件路径')
    parser.add_argument('--data-sources-dir', default='data_sources', help='数据源配置文件存放目录')
    
    # 应用配置
    parser.add_argument('--app-config', default='app_config.yaml', help='应用程序配置文件路径')
    
    # 覆盖选项
    parser.add_argument('--collection', help='向量存储集合名称')
    parser.add_argument('--persist-dir', help='向量存储持久化目录')
    
    args = parser.parse_args()
    
    print("\n=== RAG数据导入工具 ===\n")
    
    # 加载应用程序配置
    app_config = AppConfigManager(config_path=args.app_config)
    
    # 覆盖配置（如果指定）
    if args.collection:
        app_config._set_nested_value(app_config.config, ["vector_store", "default_collection"], args.collection)
    
    if args.persist_dir:
        app_config._set_nested_value(app_config.config, ["vector_store", "persist_dir"], args.persist_dir)
    
    # 执行导入
    import_data(
        app_config=app_config,
        data_config_path=args.data_config,
        data_sources_dir=args.data_sources_dir
    )
    
    print("\n数据导入完成！\n")


if __name__ == "__main__":
    main()