#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG系统交互式查询工具

这个脚本提供了一个交互式界面，用于查询已导入到向量存储中的知识库。
"""

import argparse
import datetime
import sys
import time
import os
from typing import List, Dict, Any, Optional

from data_types import Document
from embeddings import create_embedding_model
from vector_store import ChromaVectorStore
from generator import create_generator
from rag_engine import RAG
from app_config_manager import AppConfigManager


def format_citation(citation):
    """格式化引用为可读形式"""
    if "url" in citation:
        return f"[{citation['index']}] {citation['title']}: {citation['url']}"
    else:
        return f"[{citation['index']}] {citation['title']}"


def interactive_session(rag_engine, use_stream=False):
    """交互式RAG会话"""
    print("\n=== RAG交互式会话 ===")
    print("输入问题开始对话，输入'quit'或'exit'退出")
    print("输入'toggle stream'切换流式输出模式")
    print(f"流式输出: {'启用' if use_stream else '禁用'}")
    
    while True:
        # 获取用户输入
        user_input = input("\n问题> ")
        
        # 检查是否退出
        if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
            print("\n会话结束，再见！")
            break
        
        # 切换流式输出模式
        if user_input.lower() == 'toggle stream':
            use_stream = not use_stream
            print(f"流式输出已{'启用' if use_stream else '禁用'}")
            continue
        
        # 检查空输入
        if not user_input.strip():
            continue
        
        # 执行RAG查询
        start_time = datetime.datetime.now()
        result = rag_engine.query(user_input, stream=use_stream)
        
        # 输出响应
        print("\n回答:")
        
        if use_stream:
            # 流式输出
            stream_generator = result["response"]
            full_response = ""
            
            for chunk in stream_generator():
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()  # 打印一个换行
            result["response"] = full_response  # 保存完整响应用于记录
        else:
            # 非流式输出
            print(result["response"])
        
        end_time = datetime.datetime.now()
        
        # 显示检索到的文档数和引用
        doc_count = len(result["documents"])
        print(f"\n[检索到 {doc_count} 个相关文档，耗时 {(end_time-start_time).total_seconds():.2f} 秒]")
        
        # 显示引用信息
        if result.get("citations"):
            print("\n引用:")
            for citation in result["citations"]:
                print(format_citation(citation))
        
        # 是否显示检索到的文档
        if doc_count > 0:
            show_docs = input("是否显示检索到的文档? (y/n) ").lower()
            if show_docs == 'y':
                for i, doc in enumerate(result["documents"]):
                    similarity = doc.metadata.get('similarity_score', 'N/A')
                    source = doc.metadata.get('source_file', 'Unknown')
                    title = doc.metadata.get('title', 'No title')
                    url = doc.metadata.get('url', 'No URL')
                    
                    print(f"\n--- 文档 {i+1} (相似度: {similarity:.4f}) ---")
                    print(f"标题: {title}")
                    if 'url' in doc.metadata:
                        print(f"URL: {url}")
                    print(f"来源: {source}")
                    print(f"内容: {doc.content[:200]}...")


def main():
    parser = argparse.ArgumentParser(description='RAG交互式查询工具')
    # 应用配置
    parser.add_argument('--app-config', default='app_config.yaml', help='应用程序配置文件路径')
    
    # 覆盖选项
    parser.add_argument('--collection', help='向量存储集合名称')
    parser.add_argument('--persist-dir', help='向量存储持久化目录')
    parser.add_argument('--generator-type', help='生成器类型 (ollama/openai)')
    parser.add_argument('--model', help='模型名称')
    parser.add_argument('--api-url', help='API URL')
    parser.add_argument('--stream', action='store_true', help='启用流式输出')
    
    args = parser.parse_args()
    
    print("\n=== RAG交互式查询工具 ===\n")
    
    # 加载应用程序配置
    app_config = AppConfigManager(config_path=args.app_config)
    
    # 覆盖配置（如果指定）
    if args.collection:
        app_config._set_nested_value(app_config.config, ["vector_store", "default_collection"], args.collection)
    
    if args.persist_dir:
        app_config._set_nested_value(app_config.config, ["vector_store", "persist_dir"], args.persist_dir)
    
    if args.generator_type:
        app_config._set_nested_value(app_config.config, ["generator", "default_type"], args.generator_type)
    
    if args.model:
        generator_type = app_config.get("generator", "default_type", "ollama")
        app_config._set_nested_value(app_config.config, ["generator", generator_type, "default_model"], args.model)
    
    if args.api_url:
        generator_type = app_config.get("generator", "default_type", "ollama")
        app_config._set_nested_value(app_config.config, ["generator", generator_type, "api_url"], args.api_url)
    
    if args.stream:
        generator_type = app_config.get("generator", "default_type", "ollama")
        app_config._set_nested_value(app_config.config, ["generator", generator_type, "streaming"], True)
    
    # 获取配置
    collection_name = app_config.get("vector_store", "default_collection")
    persist_dir = app_config.get("vector_store", "persist_dir")
    
    # 检查配置值是否为None
    if collection_name is None:
        collection_name = "wiki_knowledge_base"
        print(f"警告: 配置中未找到vector_store.default_collection，使用默认值{collection_name}")
    
    if persist_dir is None:
        persist_dir = "./chroma_db"
        print(f"警告: 配置中未找到vector_store.persist_dir，使用默认值{persist_dir}")
    
    print(f"使用向量存储: 集合={collection_name}, 路径={persist_dir}")
    
    # 初始化嵌入模型
    embedding_model = create_embedding_model(app_config)
    
    # 初始化向量存储
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        embedding_model=embedding_model,
        persist_dir=persist_dir
    )
    
    # 初始化生成器
    generator = create_generator(app_config)
    
    # 获取RAG配置
    top_k = app_config.get("rag", "top_k", 5)
    prompt_template = app_config.get("rag", "prompt_template")
    
    # 确保值不为None
    if top_k is None:
        top_k = 5
        print(f"警告: 配置中未找到rag.top_k，使用默认值{top_k}")
    
    if prompt_template is None:
        prompt_template = """
你是一个知识丰富的AI助手。请使用以下检索到的相关信息来回答用户问题。如果检索到的信息不足以回答问题，请说明你不知道。

相关信息:
{context}

用户问题: {query}

请提供准确、全面的回答。如果你使用了包含URL的文档，请在回答中引用这些来源，格式为[文档标题](URL)。
"""
        print("警告: 配置中未找到rag.prompt_template，使用默认模板")
    
    # 初始化RAG引擎
    rag_engine = RAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        generator=generator,
        top_k=top_k,
        prompt_template=prompt_template
    )
    
    # 获取流式输出设置
    generator_type = app_config.get("generator", "default_type", "ollama")
    use_stream = app_config.get("generator", generator_type, "streaming", False)
    
    if use_stream is None:
        use_stream = False
        print(f"警告: 配置中未找到generator.{generator_type}.streaming，使用默认值False")
    
    # 启动交互式会话
    interactive_session(rag_engine, use_stream=use_stream)


if __name__ == "__main__":
    main()