#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识库后端服务 - 完整的 RAG 流水线

功能：
1. 接收前端上传的文档（PDF/Word/PPT 等）
2. 自动执行完整 RAG 流水线：
   - 文档解析 (MinerU)
   - 数据清洗 (策略模式)
   - 文本切块 (LangChain)
   - 向量存储 (Qdrant + BGE-M3)
3. 提供知识检索接口

基于 src/test/knowledge_api.py 扩展实现
"""

import os
import sys
import json
import shutil
import tempfile
import traceback
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct
)
import torch
import uuid

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # src
BASE_DIR = os.path.dirname(BASE_DIR)  # project root

# 中间数据目录
MIDDLE_DIR = os.path.join(BASE_DIR, "data", "middle")
MINERU_OUTPUT_DIR = os.path.join(MIDDLE_DIR, "mineru_output")
CLEANED_OUTPUT_DIR = os.path.join(MIDDLE_DIR, "cleaned_output")
CHUNK_OUTPUT_DIR = os.path.join(MIDDLE_DIR, "chunk_output")

# 模型和向量数据库配置
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model", "bge-m3-model")
QDRANT_LOCAL_PATH = os.path.join(BASE_DIR, "qdrant_local")
COLLECTION_NAME = "mineru_rag_collection"
VECTOR_SIZE = 1024
TOP_K = 3

# 确保目录存在
for dir_path in [MINERU_OUTPUT_DIR, CLEANED_OUTPUT_DIR, CHUNK_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print(f"项目根目录：{BASE_DIR}")
print(f"MinerU 输出目录：{MINERU_OUTPUT_DIR}")
print(f"清洗输出目录：{CLEANED_OUTPUT_DIR}")
print(f"切块输出目录：{CHUNK_OUTPUT_DIR}")
print(f"模型路径：{LOCAL_MODEL_PATH}")
print(f"Qdrant 存储路径：{QDRANT_LOCAL_PATH}")


# ==================== 响应模型 ====================
class RetrievalResult(BaseModel):
    """单个检索结果"""
    content: str
    doc_name: Optional[str] = None
    headers: Optional[str] = None
    score: float
    trust_score: float


class QueryResponse(BaseModel):
    """查询响应"""
    query: str
    results: List[RetrievalResult]


class UploadResponse(BaseModel):
    """文件上传响应"""
    status: str
    message: str
    file_name: str
    file_id: str
    stages_completed: List[str]


class PipelineStatus(BaseModel):
    """流水线状态"""
    file_id: str
    stage: str
    status: str
    message: str
    progress: int  # 0-100


# ==================== 全局变量 ====================
app = FastAPI(
    title="知识库 RAG 后端服务",
    description="完整的 RAG 流水线：文档上传 → 解析 → 清洗 → 切块 → 向量化 → 检索",
    version="2.0.0"
)

_global_client: Optional[QdrantClient] = None
_global_embeddings: Optional[HuggingFaceEmbeddings] = None

# 任务状态跟踪
task_status: Dict[str, Dict[str, Any]] = {}


# ==================== 工具函数 ====================
def get_file_md5(file_content: bytes) -> str:
    """计算文件 MD5"""
    return hashlib.md5(file_content).hexdigest()


def calculate_trust_score(score: float, max_score: float = 1.0) -> float:
    """计算信任值"""
    return round(min(max(score * score, 0.0), 1.0), 4)


def _build_qdrant_filter(filter_dict: dict) -> Optional[Filter]:
    """构建 Qdrant 过滤条件"""
    if not filter_dict:
        return None
    
    conditions = []

    def parse_condition(key, value):
        if isinstance(value, dict):
            op = list(value.keys())[0]
            val = value[op]
            if op == "$eq":
                return FieldCondition(key=key, match=MatchValue(value=val))
            elif op == "$ne":
                return FieldCondition(key=key, match=MatchValue(value=val), except_=True)
        else:
            return FieldCondition(key=key, match=MatchValue(value=value))

    if "$and" in filter_dict:
        must_conditions = []
        for cond in filter_dict["$and"]:
            for k, v in cond.items():
                must_conditions.append(parse_condition(k, v))
        return Filter(must=must_conditions)
    elif "$or" in filter_dict:
        should_conditions = []
        for cond in filter_dict["$or"]:
            for k, v in cond.items():
                should_conditions.append(parse_condition(k, v))
        return Filter(should=should_conditions)
    else:
        must_conditions = []
        for k, v in filter_dict.items():
            must_conditions.append(parse_condition(k, v))
        return Filter(must=must_conditions)


# ==================== 阶段 1: 文档解析 (MinerU) ====================
def stage1_parse_document(input_file: str, output_dir: str, file_id: str) -> Dict[str, Any]:
    """
    阶段 1: 使用 MinerU 解析文档

    采用新版 API（magic-pdf 1.x / MinerU 2.x）：
    PymuDocDataset + operators 流水线，OCR/文本模式由框架自动调度。
    旧版 magic_pdf.pipe.UNIPipe 已被移除，不可用。
    """
    print(f"\n{'='*60}")
    print(f"阶段 1: 文档解析 (MinerU)")
    print(f"{'='*60}")

    try:
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
        from magic_pdf.config.constants import MODEL_NAME

        input_path = Path(input_file)
        file_name = input_path.stem

        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在：{input_file}")

        # 读取 PDF 原始字节
        with open(input_file, "rb") as f:
            pdf_bytes = f.read()
        if not pdf_bytes:
            raise ValueError(f"PDF 文件为空：{input_file}")

        os.makedirs(output_dir, exist_ok=True)

        # 构建 Dataset
        ds = PymuDocDataset(pdf_bytes)

        # 自动判断解析方式（等价于官方 CLI 的 method='auto'）
        pdf_type = ds.classify()
        use_ocr = (pdf_type != SupportedPdfParseMethod.TXT)
        print(f"  PDF 类型：{pdf_type}（{'OCR 模式' if use_ocr else '文本模式'}）")

        # 准备图片输出目录（OCR/含图 PDF 会导出图片）
        image_dir_name = f"{file_id}_images"
        local_image_dir = Path(output_dir) / image_dir_name
        os.makedirs(local_image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(str(local_image_dir))

        # 推理 + 解析流水线
        # - layout_model=doclayout_yolo：默认 layoutlmv3 依赖 detectron2（Windows 极难装）
        # - formula_enable=False：当前环境 transformers 版本与 UniMERNet 不兼容
        infer_result = ds.apply(
            doc_analyze,
            ocr=use_ocr,
            layout_model=MODEL_NAME.DocLayout_YOLO,
            formula_enable=False,
        )
        if use_ocr:
            pipe_result = infer_result.pipe_ocr_mode(image_writer, debug_mode=False)
        else:
            pipe_result = infer_result.pipe_txt_mode(image_writer, debug_mode=False)

        # 直接获取解析结果（无需落盘中间文件）
        md_content = pipe_result.get_markdown(image_dir_name) or ""
        content_list = pipe_result.get_content_list(image_dir_name)
        if isinstance(content_list, str):
            content_list = json.loads(content_list)
        if not isinstance(content_list, list):
            content_list = []

        # 真正无内容时按失败处理，避免空结果进入下游阶段
        if not md_content.strip() and not content_list:
            raise ValueError(
                "未能从 PDF 中提取到任何文本内容。可能原因：1) PDF 为扫描版且 OCR 模型权重未就绪；"
                "2) PDF 加密或损坏。请检查 MinerU 模型是否已下载。"
            )

        # 构建输出 JSON
        # content 直接保存 MinerU 原始 content_list（每项含 type/text/page_idx 等），
        # 供 stage2 的清洗策略（FaultCaseStrategy/DeclarationStrategy）按其契约消费。
        output_data = {
            "metadata": {
                "source_file": str(input_path.absolute()),
                "file_id": file_id,
                "parse_time": datetime.now().isoformat(),
                "parser": "mineru",
                "file_name": file_name,
                "file_size": input_path.stat().st_size,
                "parse_method": "ocr" if use_ocr else "txt"
            },
            "markdown": md_content,
            "content": content_list
        }

        # 保存解析结果
        output_json_path = Path(output_dir) / f"{file_id}.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 解析完成：{file_name}")
        print(f"  输出文件：{output_json_path}")
        print(f"  内容块数：{len(output_data['content'])}")

        return {
            "status": "success",
            "output_file": str(output_json_path),
            "pages": len(output_data["content"]),
            "message": "Parsing completed successfully"
        }

    except Exception as e:
        error_msg = f"解析失败：{str(e)}"
        print(f"✗ {error_msg}")
        traceback.print_exc()

        return {
            "status": "error",
            "message": error_msg
        }


# ==================== 阶段 2: 数据清洗 ====================
def stage2_clean_data(json_path: str, output_dir: str, file_id: str) -> Dict[str, Any]:
    """
    阶段 2: 数据清洗与结构重组
    导入本地策略模块
    """
    print(f"\n{'='*60}")
    print(f"阶段 2: 数据清洗与结构重组")
    print(f"{'='*60}")
    
    try:
        # 动态导入策略模块
        sys.path.insert(0, os.path.join(BASE_DIR, "src", "clean"))
        from faultCaseStrategy import FaultCaseStrategy
        from declarationStrategy import DeclarationStrategy

        # 加载 stage1 产物：{metadata, markdown, content}
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 方案 A：从 content 字段取 MinerU 原始 content_list（策略契约要求的格式）
        if isinstance(raw_data, dict):
            blocks = raw_data.get("content", [])
            file_name = raw_data.get("metadata", {}).get("file_name") or os.path.basename(json_path)
        elif isinstance(raw_data, list):
            # 兼容直接落盘的裸 content_list
            blocks = raw_data
            file_name = os.path.basename(json_path)
        else:
            raise ValueError(f"无法识别的 stage1 产物结构：{type(raw_data)}")

        # 展平数据结构（兼容 1D / 2D 数组，与 clean.py 一致）
        all_blocks = []
        if blocks:
            if isinstance(blocks[0], dict):
                all_blocks = blocks
            elif isinstance(blocks[0], list):
                for page_idx, page_blocks in enumerate(blocks):
                    for block in page_blocks:
                        if "page_idx" not in block and "page_idx " not in block:
                            block["page_idx"] = page_idx
                        all_blocks.append(block)

        # 智能路由：按文件名嗅探文档类型，默认 fault_case
        def detect_doc_type(name: str) -> str:
            if "质量案例" in name or "案例" in name:
                return "fault_case"
            if "申报" in name or "报告" in name:
                return "declaration"
            return "fault_case"

        strategies = {
            "fault_case": FaultCaseStrategy(),
            "declaration": DeclarationStrategy()
        }
        doc_type = detect_doc_type(file_name)
        strategy = strategies.get(doc_type, strategies["fault_case"])
        print(f"  文档类型：{doc_type}（策略：{strategy.__class__.__name__}）")

        # 执行清洗
        raw_docs = strategy.parse(all_blocks, file_name, doc_type)
        
        # 生成 block_id，并写入 metadata 以便沿 stage3 -> stage4 流转（用于去重/增量更新）
        for doc in raw_docs:
            content_hash = hashlib.md5(doc["content"].encode('utf-8')).hexdigest()[:8]
            block_id = f"{file_id}_{content_hash}"
            doc["block_id"] = block_id
            doc["metadata"]["block_id"] = block_id
        
        # 保存清洗结果
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{file_id}_clean.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(raw_docs, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 清洗完成")
        print(f"  输出文件：{output_path}")
        print(f"  文档块数量：{len(raw_docs)}")
        
        return {
            "status": "success",
            "output_file": output_path,
            "doc_count": len(raw_docs),
            "message": "Cleaning completed successfully"
        }
        
    except Exception as e:
        error_msg = f"清洗失败：{str(e)}"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": error_msg
        }


# ==================== 阶段 3: 文本切块 ====================
def stage3_chunk_documents(json_path: str, output_dir: str, file_id: str) -> Dict[str, Any]:
    """
    阶段 3: LangChain 智能文本切块
    """
    print(f"\n{'='*60}")
    print(f"阶段 3: 文本切块 (LangChain)")
    print(f"{'='*60}")
    
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # 加载清洗后的数据
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 智能切块
        final_docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
        
        for item in raw_data:
            metadata = item["metadata"]
            content = item["content"]
            chunk_type = metadata.get("chunk_type", "text")
            
            if chunk_type == "table":
                # 表格不切碎
                final_docs.append(Document(page_content=content, metadata=metadata))
            else:
                # 正文进行智能切分
                splits = text_splitter.split_text(content)
                for split in splits:
                    final_docs.append(Document(page_content=split, metadata=metadata.copy()))
        
        # 保存切块结果
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{file_id}_chunk.json")
        
        serializable_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in final_docs
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 切块完成")
        print(f"  输出文件：{output_path}")
        print(f"  切片数量：{len(final_docs)}")
        
        return {
            "status": "success",
            "output_file": output_path,
            "chunk_count": len(final_docs),
            "message": "Chunking completed successfully"
        }
        
    except Exception as e:
        error_msg = f"切块失败：{str(e)}"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": error_msg
        }


# ==================== 阶段 4: 向量存储 ====================
def stage4_embed_and_store(json_path: str, client: QdrantClient, 
                           embeddings: HuggingFaceEmbeddings,
                           collection_name: str, file_id: str) -> Dict[str, Any]:
    """
    阶段 4: 向量化并存入 Qdrant
    """
    print(f"\n{'='*60}")
    print(f"阶段 4: 向量存储 (Qdrant + BGE-M3)")
    print(f"{'='*60}")
    
    try:
        from langchain_core.documents import Document

        # 加载切块数据
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 转换为 Document 对象并清洗 metadata
        docs = []
        for item in raw_data:
            clean_meta = {}
            for k, v in item['metadata'].items():
                if isinstance(v, (list, dict)):
                    clean_meta[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    clean_meta[k] = ""
                else:
                    clean_meta[k] = v
            
            # 添加 file_id 到 metadata
            clean_meta['file_id'] = file_id
            
            docs.append(Document(page_content=item['page_content'], metadata=clean_meta))
        
        print(f"加载 {len(docs)} 个文档块")
        
        # 分批向量化并存储
        batch_size = 100
        total_upserted = 0
        
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            texts = [d.page_content for d in batch_docs]
            metadatas = [d.metadata for d in batch_docs]
            
            # 生成唯一 ID
            ids = []
            for idx, doc in enumerate(batch_docs):
                bid = doc.metadata.get("block_id")
                if bid and isinstance(bid, (int, str)):
                    try:
                        ids.append(int(bid))
                    except (ValueError, TypeError):
                        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                        ids.append(str(uuid.UUID(content_hash[:32])))
                else:
                    content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                    ids.append(str(uuid.UUID(content_hash[:32])))
            
            # 计算向量
            print(f"  正在计算第 {i+1}-{min(i+batch_size, len(docs))} 条向量...")
            batch_embeddings = embeddings.embed_documents(texts)
            
            # 构建 PointStruct
            points = []
            for idx, (vec, text, meta) in enumerate(zip(batch_embeddings, texts, metadatas)):
                payload = {
                    "page_content": text,
                    **meta
                }
                points.append(PointStruct(id=ids[idx], vector=vec, payload=payload))
            
            # Upsert 到 Qdrant
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            total_upserted += len(batch_docs)
        
        print(f"✓ 向量存储完成")
        print(f"  入库数据量：{total_upserted}")
        
        return {
            "status": "success",
            "upserted_count": total_upserted,
            "message": "Embedding and storage completed successfully"
        }
        
    except Exception as e:
        error_msg = f"向量存储失败：{str(e)}"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": error_msg
        }


# ==================== 完整流水线 ====================
def run_full_pipeline(file_path: str, file_id: str, file_name: str):
    """
    执行完整的 RAG 流水线
    """
    task_status[file_id] = {
        "stage": "initializing",
        "status": "running",
        "message": "开始处理",
        "progress": 0
    }
    
    try:
        # 阶段 1: 文档解析
        task_status[file_id]["stage"] = "parsing"
        task_status[file_id]["progress"] = 10
        task_status[file_id]["message"] = "正在解析文档..."
        
        parse_result = stage1_parse_document(file_path, MINERU_OUTPUT_DIR, file_id)
        if parse_result["status"] != "success":
            raise Exception(parse_result.get("message", "解析失败"))
        
        # 阶段 2: 数据清洗
        task_status[file_id]["stage"] = "cleaning"
        task_status[file_id]["progress"] = 40
        task_status[file_id]["message"] = "正在清洗数据..."
        
        clean_input = parse_result["output_file"]
        clean_result = stage2_clean_data(clean_input, CLEANED_OUTPUT_DIR, file_id)
        if clean_result["status"] != "success":
            raise Exception(clean_result.get("message", "清洗失败"))
        
        # 阶段 3: 文本切块
        task_status[file_id]["stage"] = "chunking"
        task_status[file_id]["progress"] = 70
        task_status[file_id]["message"] = "正在切分文本..."
        
        chunk_input = clean_result["output_file"]
        chunk_result = stage3_chunk_documents(chunk_input, CHUNK_OUTPUT_DIR, file_id)
        if chunk_result["status"] != "success":
            raise Exception(chunk_result.get("message", "切块失败"))
        
        # 阶段 4: 向量存储
        task_status[file_id]["stage"] = "embedding"
        task_status[file_id]["progress"] = 90
        task_status[file_id]["message"] = "正在向量化存储..."
        
        global _global_client, _global_embeddings
        if _global_client is None or _global_embeddings is None:
            raise Exception("向量数据库或嵌入模型未初始化")
        
        embed_input = chunk_result["output_file"]
        embed_result = stage4_embed_and_store(
            embed_input, 
            _global_client, 
            _global_embeddings, 
            COLLECTION_NAME, 
            file_id
        )
        if embed_result["status"] != "success":
            raise Exception(embed_result.get("message", "向量存储失败"))
        
        # 完成
        task_status[file_id]["stage"] = "completed"
        task_status[file_id]["status"] = "success"
        task_status[file_id]["progress"] = 100
        task_status[file_id]["message"] = f"处理完成，共入库 {embed_result['upserted_count']} 条数据"
        
        print(f"\n{'='*60}")
        print(f"✓ 完整流水线执行成功！")
        print(f"  文件 ID: {file_id}")
        print(f"  文件名：{file_name}")
        print(f"  入库数据量：{embed_result['upserted_count']}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        task_status[file_id]["stage"] = "failed"
        task_status[file_id]["status"] = "error"
        task_status[file_id]["message"] = str(e)
        task_status[file_id]["progress"] = 0
        
        print(f"\n{'='*60}")
        print(f"✗ 流水线执行失败！")
        print(f"  文件 ID: {file_id}")
        print(f"  错误信息：{str(e)}")
        print(f"{'='*60}\n")
        traceback.print_exc()


# ==================== FastAPI 事件 ====================
@app.on_event("startup")
async def startup_event():
    """服务启动时初始化"""
    global _global_client, _global_embeddings
    
    try:
        print("="*60)
        print("知识库 RAG 后端服务启动中...")
        print("="*60)
        
        # 1. 初始化 Qdrant 客户端
        os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
        _global_client = QdrantClient(path=QDRANT_LOCAL_PATH)
        print(f"✓ Qdrant 客户端已初始化：{QDRANT_LOCAL_PATH}")
        
        # 2. 确保集合存在
        try:
            _global_client.get_collection(collection_name=COLLECTION_NAME)
            print(f"✓ 集合 '{COLLECTION_NAME}' 已存在，复用")
        except Exception:
            print(f"→ 集合 '{COLLECTION_NAME}' 不存在，正在创建...")
            _global_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ 集合 '{COLLECTION_NAME}' 创建成功")
        
        # 3. 初始化 BGE-M3 模型
        print(f"→ 正在加载 BGE-M3 模型：{LOCAL_MODEL_PATH} ...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  检测到设备：{device}")
        
        model_kwargs = {
            'device': device,
            'trust_remote_code': True
        }
        encode_kwargs = {
            'normalize_embeddings': True
        }
        
        _global_embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # 测试模型
        test_vec = _global_embeddings.embed_query("测试")
        print(f"✓ 模型加载成功！向量维度：{len(test_vec)}")
        
        print("="*60)
        print("✓ 服务启动完成！")
        print("="*60)
        
    except Exception as e:
        print("\n✗ 服务启动失败！")
        traceback.print_exc()
        raise RuntimeError(f"服务启动失败：{str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理资源"""
    global _global_client
    
    if _global_client is not None:
        try:
            _global_client.close()
            print("✓ Qdrant 客户端已关闭")
        except Exception:
            pass


# ==================== API 接口 ====================
@app.post(
    "/upload_document",
    response_model=UploadResponse,
    summary="上传文档并执行完整 RAG 流水线",
    description="接收前端上传的文档，自动执行：解析 → 清洗 → 切块 → 向量存储"
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="要上传的文档文件 (PDF/Word/PPT 等)")
):
    """
    上传文档并执行完整 RAG 流水线
    
    - **file**: 文档文件（支持 PDF、Word、PPT 等格式）
    
    返回：
    - **status**: 处理状态
    - **file_id**: 文件唯一标识
    - **stages_completed**: 已完成的处理阶段
    """
    # 验证文件扩展名
    allowed_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件格式：{file_ext}，支持的格式：{', '.join(allowed_extensions)}"
        )
    
    # 生成文件 ID
    file_content = await file.read()
    file_md5 = get_file_md5(file_content)
    file_id = f"{file_md5}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 保存临时文件
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_file_path, 'wb') as f:
        f.write(file_content)
    
    print(f"\n{'='*60}")
    print(f"收到上传文件：{file.filename}")
    print(f"文件 ID: {file_id}")
    print(f"文件大小：{len(file_content)} bytes")
    print(f"{'='*60}")
    
    # 在后台执行完整流水线
    background_tasks.add_task(run_full_pipeline, temp_file_path, file_id, file.filename)
    
    return UploadResponse(
        status="processing",
        message="文件已接收，正在后台处理中",
        file_name=file.filename,
        file_id=file_id,
        stages_completed=[]
    )


@app.get(
    "/pipeline_status/{file_id}",
    response_model=PipelineStatus,
    summary="查询流水线处理状态",
    description="根据 file_id 查询文档处理进度"
)
async def get_pipeline_status(file_id: str):
    """查询流水线处理状态"""
    if file_id not in task_status:
        raise HTTPException(status_code=404, detail=f"未找到任务：{file_id}")
    
    status = task_status[file_id]
    return PipelineStatus(
        file_id=file_id,
        stage=status["stage"],
        status=status["status"],
        message=status["message"],
        progress=status["progress"]
    )


@app.get(
    "/query",
    response_model=QueryResponse,
    summary="知识库检索接口",
    description="接收用户问题，返回 TOP-K 相关知识点及信任值"
)
async def query_knowledge_base(
    question: str = Query(..., description="用户提出的问题", min_length=1, max_length=1000),
    top_k: int = Query(default=TOP_K, description="返回结果数量", ge=1, le=10),
    filter_json: Optional[str] = Query(default=None, description="可选的元数据过滤条件（JSON 格式）")
):
    """
    知识库问答检索接口
    
    - **question**: 用户提出的问题
    - **top_k**: 返回的结果数量（默认 3，最大 10）
    - **filter_json**: 可选的元数据过滤条件，JSON 格式
    
    返回 TOP-K 相关信息及其信任值
    """
    if _global_client is None or _global_embeddings is None:
        raise HTTPException(status_code=503, detail="服务未完全初始化，请稍后重试")
    
    try:
        # 将查询向量化
        query_vector = _global_embeddings.embed_query(question)
        
        # 构建过滤条件
        qdrant_filter = None
        if filter_json:
            try:
                filter_dict = json.loads(filter_json)
                qdrant_filter = _build_qdrant_filter(filter_dict)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"filter_json 格式错误：{str(e)}")
        
        # 执行搜索
        search_result = _global_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # 构建响应结果
        results = []
        for point in search_result.points:
            score = point.score
            payload = point.payload
            
            content = payload.get('page_content', '') if payload else ''
            doc_name = payload.get('doc_name') if payload else None
            headers = payload.get('headers') if payload else None
            
            trust_score = calculate_trust_score(score)
            
            results.append(RetrievalResult(
                content=content,
                doc_name=doc_name,
                headers=headers,
                score=round(score, 4),
                trust_score=trust_score
            ))
        
        return QueryResponse(
            query=question,
            results=results
        )
        
    except Exception as e:
        print(f"\n检索过程发生错误！")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")


@app.get("/health", summary="健康检查接口")
async def health_check():
    """健康检查接口"""
    status = {
        "status": "healthy",
        "client_ready": _global_client is not None,
        "embeddings_ready": _global_embeddings is not None,
        "pipeline_active": len(task_status) > 0
    }
    return status


@app.get("/stats", summary="系统统计信息")
async def get_stats():
    """获取系统统计信息"""
    stats = {
        "active_tasks": len(task_status),
        "collections": []
    }
    
    if _global_client:
        try:
            collections = _global_client.get_collections()
            stats["collections"] = [
                {
                    "name": col.name,
                    "points_count": col.points_count
                }
                for col in collections.collections
            ]
        except Exception:
            pass
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
