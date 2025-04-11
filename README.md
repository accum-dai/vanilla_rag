# RAG知识库系统

## 项目简介

RAG知识库系统是一个基于检索增强生成（Retrieval-Augmented Generation, RAG）技术的智能问答系统。该系统通过向量数据库存储和检索文档，结合大型语言模型，能够回答基于知识库的问题，并提供准确的引用来源。

### 主要特点

- **批量数据处理**：优化的内存管理，能够处理大规模文档
- **GPU加速**：自动检测并利用GPU加速嵌入向量计算
- **多模型支持**：支持Ollama和OpenAI等多种大型语言模型
- **配置驱动**：通过YAML和.env文件进行灵活配置
- **数据源追踪**：自动记录导入状态，支持增量更新
- **流式输出**：实时生成回答，提升用户体验

## 系统架构

系统分为两个主要组件：

1. **数据导入工具**（import_data.py）：将数据源配置中的文档导入到向量数据库
2. **交互式查询工具**（interactive.py）：连接向量数据库和语言模型进行问答

## 安装指南

### 环境要求

- Python 3.8+
- CUDA支持（推荐，用于GPU加速）
- 足够的存储空间（视知识库大小而定）

### 安装步骤

1. 克隆仓库或下载源代码：

```bash
git clone https://github.com/yourusername/rag-knowledge-base.git
cd rag-knowledge-base
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备配置文件：

```bash

# 创建数据源目录
mkdir -p data_sources
```

## 配置说明

### 应用配置（app_config.yaml）

```yaml
# 生成器设置
generator:
  default_type: "ollama"  # 默认生成器类型
  
  # Ollama设置
  ollama:
    api_url: "http://localhost:11434/api/generate"
    default_model: "qwen:32b"
    streaming: true
  
  # OpenAI设置
  openai:
    api_url: "https://api.openai.com/v1/chat/completions"
    default_model: "gpt-3.5-turbo"
    temperature: 0.7
    max_tokens: 1000
    streaming: true

# 向量存储设置
vector_store:
  persist_dir: "./chroma_db"
  default_collection: "wiki_knowledge_base"

# 嵌入模型设置
embedding:
  default_provider: "sentence_transformer"
  
  sentence_transformer:
    model_name: "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: 32
    device: null  # null表示自动检测

# 文本分割器设置
text_splitter:
  chunk_size: 500
  overlap: 50

# RAG设置
rag:
  top_k: 5
  prompt_template: |
    你是一个知识丰富的AI助手...
```

### 敏感信息配置（.env）

用于存储API密钥等敏感信息：

```
# API密钥
OPENAI_API_KEY=your_openai_api_key
OLLAMA_API_KEY=your_ollama_api_key
```

### 数据源配置（data_sources/data_sources.yaml）

```yaml
data_sources:
  - loader_type: WikiCSVDataLoader
    source_path: /path/to/your/data.csv
    id_prefix: wiki_
    status: unprocessed
    created_at: '2025-04-10T14:30:00.000000'
    last_updated: '2025-04-10T14:30:00.000000'
    metadata:
      description: "Wikipedia data sample"
```

## 使用方法

### 数据导入

导入数据源配置中的文档到向量数据库：

```bash
python import_data.py --data-config data_sources/data_sources.yaml
```

参数说明：
- `--data-config`: 数据源配置文件路径（默认：data_sources.yaml）
- `--data-sources-dir`: 数据源配置文件目录（默认：data_sources）
- `--app-config`: 应用配置文件路径（默认：app_config.yaml）
- `--collection`: 向量存储集合名称（覆盖配置）
- `--persist-dir`: 向量存储持久化目录（覆盖配置）

### 交互式查询

启动交互式问答界面：

```bash
python interactive.py
```

参数说明：
- `--app-config`: 应用配置文件路径（默认：app_config.yaml）
- `--collection`: 向量存储集合名称（覆盖配置）
- `--persist-dir`: 向量存储持久化目录（覆盖配置）
- `--generator-type`: 生成器类型 (ollama/openai)
- `--model`: 模型名称
- `--api-url`: API URL
- `--stream`: 启用流式输出

交互式命令：
- 直接输入问题进行查询
- 输入 `toggle stream` 切换流式输出模式
- 输入 `quit`、`exit`、`q` 或 `bye` 退出会话

## 文件结构

```
rag-knowledge-base/
├── app_config.yaml         # 应用配置文件
├── .env                    # 环境变量（API密钥等）
├── data_sources/           # 数据源配置目录
│   └── data_sources.yaml   # 数据源配置文件
├── chroma_db/              # 向量数据库目录
├── import_data.py          # 数据导入工具
├── interactive.py          # 交互式查询工具
├── app_config_manager.py   # 配置管理器
├── config_manager.py       # 数据源配置管理器
├── data_processor.py       # 数据处理模块
├── data_types.py           # 数据类型定义
├── embeddings.py           # 嵌入模型模块
├── generator.py            # 生成器模块
├── rag_engine.py           # RAG引擎
├── vector_store.py         # 向量存储模块
└── requirements.txt        # 依赖清单
```

## 添加新的数据源

1. 编辑 `data_sources/data_sources.yaml` 文件，添加新的数据源：

```yaml
data_sources:
  - loader_type: WikiCSVDataLoader
    source_path: /path/to/new/data.csv
    id_prefix: new_source_
    status: unprocessed
    created_at: '2025-04-10T14:30:00.000000'
    last_updated: '2025-04-10T14:30:00.000000'
    metadata:
      description: "New data source"
```

2. 运行数据导入工具：

```bash
python import_data.py
```

系统会自动识别并处理所有状态为 `unprocessed` 的数据源。处理完成后，数据源状态会自动更新为 `processed`，并记录处理时间和文档数量。

## 数据处理流程

1. **加载配置**：从 YAML 文件加载数据源和应用配置
2. **批量处理**：分批加载和处理大型 CSV 文件，避免内存溢出
3. **文本分割**：将长文本分割为适合向量化的小块
4. **嵌入计算**：使用嵌入模型计算文本块的向量表示
5. **向量存储**：将文本块及其向量存储到向量数据库
6. **状态更新**：更新数据源处理状态，生成带时间戳的备份配置文件

## 查询流程

1. **用户输入**：接收用户问题
2. **向量检索**：将问题转换为向量并在向量数据库中检索相关文档
3. **上下文构建**：将检索到的文档组织成结构化上下文
4. **生成回答**：将问题和上下文发送给语言模型生成回答
5. **引用显示**：显示回答及其引用来源

## 常见问题

### 内存溢出

**问题**：处理大文件时出现内存溢出 `Killed` 错误

**解决方案**：
- 系统已优化为批处理模式，分块加载和处理文件
- 可以调整 `WikiCSVDataLoader` 的 `chunk_size` 参数
- 确保定期持久化向量存储

### 配置读取问题

**问题**：配置值读取为 `None`

**解决方案**：
- 检查 YAML 文件格式是否正确
- 系统已添加安全检查，会使用默认值并显示警告
- 确保缩进和格式符合 YAML 规范

### GPU 加速

**问题**：如何启用 GPU 加速？

**解决方案**：
- 系统会自动检测 GPU
- 可以在配置中指定 `device` 参数，如 `cuda:0`
- 确保已安装 PyTorch CUDA 版本

## 扩展指南

### 添加新的生成器

1. 在 `generator.py` 中创建新的生成器类，继承自 `Generator` 基类
2. 实现 `generate` 和 `from_config` 方法
3. 在 `create_generator` 函数中添加支持
4. 在 `app_config.yaml` 中添加配置部分

### 添加新的嵌入模型

1. 在 `embeddings.py` 中创建新的嵌入模型类，继承自 `EmbeddingModel` 基类
2. 实现所需方法
3. 在 `create_embedding_model` 函数中添加支持
4. 在 `app_config.yaml` 中添加配置部分

### 添加新的数据加载器

1. 在 `data_processor.py` 中创建新的加载器类，继承自 `DataLoader` 基类
2. 实现 `load` 方法
3. 在 `import_data_source` 函数中添加支持
4. 在数据源配置中使用新的 `loader_type`

## 性能优化

1. **批量处理**：使用生成器模式处理大型文件
2. **GPU加速**：利用GPU进行向量计算
3. **并行处理**：支持多线程处理多个数据源
4. **增量更新**：只处理未处理的数据源

## 版权和许可

本项目采用 MIT 许可证。