# MinerU RAG 知识库检索系统

基于 MinerU 文档解析、BGE-M3 嵌入模型和向量数据库（Chroma/Qdrant）构建的知识库检索问答系统。

## 项目概述

本项目实现了一个完整的 RAG（Retrieval-Augmented Generation）知识库检索流水线，支持从原始文档到向量存储再到 API 服务的全流程处理。系统能够智能识别不同类型的文档（如故障案例、申报书等），进行数据清洗、结构重组、智能分块、向量化存储，并提供 RESTful API 接口进行知识检索。

## 核心功能

### 1. 数据清洗与结构重组 (`src/clean/`)
- **智能文档类型识别**：根据文件路径自动识别文档类型（故障案例/申报书等）
- **策略模式解析**：
  - `FaultCaseStrategy`：专门处理质量故障案例文档，提取问题分类、案例名称等业务属性
  - `DeclarationStrategy`：处理申报书文档，从表格中提取项目名称、承担单位等元数据
- **数据结构扁平化**：兼容 MinerU 输出的 1D 和 2D JSON 数组格式
- **统一元数据模型**：所有文档输出统一的 metadata 格式，支持业务属性扩展

### 2. 智能文本切块 (`src/chunk/`)
- 基于 LangChain 的 `RecursiveCharacterTextSplitter` 进行智能分块
- **表格保护策略**：表格内容不被切碎，整体作为一个 Document
- **正文优化切分**：按中文标点符号（。！？等）进行语义切分
- 支持自定义 chunk_size 和 chunk_overlap 参数

### 3. 向量化存储 (`src/embedding/`)
提供两种向量数据库实现：

#### Chroma 版本 (`embedding.py`)
- 使用 BGE-M3 本地模型计算 1024 维向量
- 支持余弦相似度检索
- 批量 upsert 机制，防止 OOM
- 支持 Metadata 过滤检索

#### Qdrant 版本 (`embedding_qdrant.py`)
- 本地文件模式，无需 Docker
- 支持更灵活的 payload 数据结构
- 提供数据查看调试功能
- 支持复杂的过滤条件组合

### 4. API 服务 (`src/test/knowledge_api.py`)
- 基于 FastAPI 构建的 RESTful 接口
- **检索接口** `/query`：
  - 支持语义检索
  - 支持 Metadata 过滤
  - 返回信任值评分（基于相似度平方计算）
- **健康检查接口** `/health`
- 服务启动时自动加载模型和知识库

## 项目结构

```
/workspace
├── src/
│   ├── clean/                      # 数据清洗模块
│   │   ├── clean.py                # 主控 Pipeline 和智能路由
│   │   ├── baseParseStrategy.py    # 策略基类和统一元数据模型
│   │   ├── faultCaseStrategy.py    # 故障案例解析策略
│   │   └── declarationStrategy.py  # 申报书解析策略
│   │
│   ├── chunk/                      # 文本切块模块
│   │   └── chunk.py                # LangChain 智能切片
│   │
│   ├── embedding/                  # 向量化存储模块
│   │   ├── embedding.py            # Chroma 版本
│   │   └── embedding_qdrant.py     # Qdrant 版本
│   │
│   └── test/                       # 测试和服务模块
│       ├── knowledge_api.py        # FastAPI 检索服务
│       ├── knwodlege_query.py      # 检索测试脚本
│       ├── view_chroma.py          # Chroma 数据查看工具
│       └── export_to_html.py       # 数据导出工具
│
├── model/                          # 本地模型目录（需自行下载）
│   └── bge-m3-model/               # BGE-M3 嵌入模型
│
├── data/middle/                    # 中间数据目录
│   ├── mineru_output/              # MinerU 原始输出
│   ├── cleaned_output/             # 清洗后数据
│   └── chunk_output/               # 切块后数据
│
├── chroma_db/                      # Chroma 数据库持久化目录
└── qdrant_local/                   # Qdrant 数据库持久化目录
```

## 技术栈

- **文档解析**：MinerU
- **嵌入模型**：BGE-M3 (HuggingFace)
- **向量数据库**：Chroma / Qdrant
- **Web 框架**：FastAPI
- **深度学习框架**：PyTorch
- **文本处理**：LangChain, markdownify, BeautifulSoup

## 快速开始

### 环境准备

1. **安装依赖**
```bash
pip install langchain-core langchain-huggingface langchain-chroma langchain-text-splitters
pip install qdrant-client chromadb fastapi uvicorn pydantic
pip install beautifulsoup4 markdownify torch sentence-transformers
```

2. **下载 BGE-M3 模型**
```bash
# 将模型下载到 project/model/bge-m3-model/ 目录
# 可从 HuggingFace 下载：https://huggingface.co/BAAI/bge-m3
```

### 使用流程

#### 步骤 1: 数据清洗
```bash
cd src/clean
python clean.py
```
- 输入：`data/middle/mineru_output/` 下的 MinerU JSON 文件
- 输出：`data/middle/cleaned_output/` 下的清洗后 JSON 文件

#### 步骤 2: 文本切块
```bash
cd src/chunk
python chunk.py
```
- 输入：`data/middle/cleaned_output/` 下的清洗数据
- 输出：`data/middle/chunk_output/` 下的切块数据

#### 步骤 3: 向量化存储

**使用 Chroma:**
```bash
cd src/embedding
python embedding.py
```

**使用 Qdrant:**
```bash
cd src/embedding
python embedding_qdrant.py
```

#### 步骤 4: 启动 API 服务
```bash
cd src/test
python knowledge_api.py
# 或
uvicorn knowledge_api:app --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 Swagger UI 文档

### API 使用示例

```bash
# 简单检索
curl "http://localhost:8000/query?question=1533B发生过哪些问题"

# 带过滤条件的检索
curl "http://localhost:8000/query?question=处理措施有哪些？&filter_json={\"$and\":[{\"doc_type\":{\"$eq\":\"fault_case\"}},{\"问题分类\":{\"$eq\":\"操作\"}}]}"

# 健康检查
curl http://localhost:8000/health
```

## 核心特性

### 1. 智能文档路由
系统根据文件路径自动识别文档类型并应用相应的解析策略：
- 包含"质量案例"路径 → FaultCaseStrategy
- 包含"报告"路径 → DeclarationStrategy

### 2. 业务属性提取
- **故障案例**：自动提取问题分类、案例名称等全局属性
- **申报书**：从首页表格提取项目名称、承担单位、负责人等信息

### 3. 唯一 ID 生成
- 优先使用阶段二生成的 `block_id`
- 兜底方案：文件名 + 内容 MD5 哈希

### 4. Metadata 类型清洗
自动处理 Chroma/Qdrant 对 metadata 的类型限制：
- list/dict 类型序列化为 JSON 字符串
- None 值转换为空字符串

### 5. 信任值评分
基于相似度分数计算信任值：`trust_score = score²`
强调高相似度的可信度，范围 [0, 1]

## 配置说明

在 `embedding.py` 和 `embedding_qdrant.py` 中可配置：

```python
# 模型路径
LOCAL_MODEL_PATH = "path/to/bge-m3-model"

# 数据存储路径
CHROMA_PERSIST_DIR = "path/to/chroma_db"
QDRANT_LOCAL_PATH = "path/to/qdrant_local"

# 集合名称
COLLECTION_NAME = "mineru_rag_collection"

# 向量维度（BGE-M3 默认 1024）
VECTOR_SIZE = 1024
```

## 注意事项

1. **GPU 加速**：系统会自动检测 CUDA 设备，如有 NVIDIA GPU 可加速向量化计算
2. **内存管理**：大批量数据处理时使用 batch 模式，防止 OOM
3. **模型路径**：确保 BGE-M3 模型路径正确，且包含完整模型文件
4. **数据兼容性**：支持 MinerU 不同版本的输出格式（带空格或不带空格的 key）

## 测试工具

- `view_chroma.py`：查看 Chroma 数据库内容
- `knwodlege_query.py`：命令行检索测试
- `export_to_html.py`：导出数据为 HTML 格式

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request！
