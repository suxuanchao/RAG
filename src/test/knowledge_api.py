import os
import traceback
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition, MatchValue
)
import torch

# ================= 配置类 =================
class Config:
    """配置管理"""
    # 自动计算项目根目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(BASE_DIR)
    BASE_DIR = os.path.dirname(BASE_DIR)
    
    # 本地 BGE-M3 模型路径
    LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model", "bge-m3-model")
    
    # Qdrant 本地存储路径
    QDRANT_LOCAL_PATH = os.path.join(BASE_DIR, "qdrant_local")
    COLLECTION_NAME = "mineru_rag_collection"
    VECTOR_SIZE = 1024
    
    # 检索返回的 top-k 数量
    TOP_K = 3


# ================= 响应模型 =================
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


# ================= 全局变量 =================
app = FastAPI(
    title="知识库问答检索服务",
    description="基于 BGE-M3 嵌入模型和 Qdrant 向量数据库的知识库检索 API",
    version="1.0.0"
)

# 全局单例
_global_client: Optional[QdrantClient] = None
_global_embeddings: Optional[HuggingFaceEmbeddings] = None
_config: Optional[Config] = None


def _build_qdrant_filter(filter_dict: dict) -> Optional[Filter]:
    """
    将 Chroma 风格的 filter_dict 转换为 Qdrant 的 Filter 对象
    """
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


def init_qdrant_client(qdrant_path: str) -> QdrantClient:
    """初始化 Qdrant 客户端（本地文件模式）"""
    os.makedirs(qdrant_path, exist_ok=True)
    client = QdrantClient(path=qdrant_path)
    print(f"使用 Qdrant 本地文件模式，数据存储于: {qdrant_path}")
    return client


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024):
    """检查并创建 Collection（如果不存在）"""
    try:
        client.get_collection(collection_name=collection_name)
        print(f"集合 '{collection_name}' 已存在，复用。")
    except Exception:
        print(f"集合 '{collection_name}' 不存在，正在创建...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"集合 '{collection_name}' 创建成功（维度: {vector_size}, 距离: Cosine）")


def init_local_bge_m3(model_path: str) -> HuggingFaceEmbeddings:
    """加载本地路径下的 BGE-M3 模型"""
    print(f"正在加载本地 BGE-M3 模型: {model_path} ...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"检测到 PyTorch 可用设备: {device}")

    model_kwargs = {
        'device': device,
        'trust_remote_code': True
    }
    encode_kwargs = {
        'normalize_embeddings': True
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 测试模型是否加载成功
    test_vec = embeddings.embed_query("测试")
    print(f"模型加载成功！向量维度: {len(test_vec)} (BGE-M3 默认为 1024 维)")
    return embeddings


def calculate_trust_score(score: float, max_score: float = 1.0) -> float:
    """
    根据相似度分数计算信任值
    使用 sigmoid 函数将余弦相似度映射到 0-1 范围
    """
    # 余弦相似度范围是 [-1, 1]，但归一化后通常是 [0, 1]
    # 这里直接使用相似度作为基础，应用一个简单的转换
    # 信任值 = score^2，强调高相似度的可信度
    return round(min(max(score * score, 0.0), 1.0), 4)


@app.on_event("startup")
async def startup_event():
    """服务启动时自动加载模型和知识库"""
    global _global_client, _global_embeddings, _config
    
    try:
        print("="*50)
        print("知识库问答检索服务启动中...")
        print("="*50)
        
        # 初始化配置
        _config = Config()
        
        print(f"当前脚本所在绝对目录: {_config.BASE_DIR}")
        print(f"模型路径: {_config.LOCAL_MODEL_PATH}")
        print(f"Qdrant 本地存储路径: {_config.QDRANT_LOCAL_PATH}")
        
        # 1. 初始化 Qdrant 客户端
        _global_client = init_qdrant_client(_config.QDRANT_LOCAL_PATH)
        
        # 2. 确保集合存在
        ensure_collection(_global_client, _config.COLLECTION_NAME, _config.VECTOR_SIZE)
        
        # 3. 初始化本地 BGE-M3 模型
        _global_embeddings = init_local_bge_m3(_config.LOCAL_MODEL_PATH)
        
        print("="*50)
        print("服务启动完成！")
        print("="*50)
        
    except Exception as e:
        print("\n服务启动失败！详细堆栈信息如下：")
        traceback.print_exc()
        raise RuntimeError(f"服务启动失败: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理资源"""
    global _global_client
    
    if _global_client is not None:
        try:
            _global_client.close()
            print("Qdrant 客户端已关闭")
        except Exception:
            pass


@app.get(
    "/query",
    response_model=QueryResponse,
    summary="知识库检索接口",
    description="接收用户问题，返回 TOP-3 相关知识点及信任值"
)
async def query_knowledge_base(
    question: str = Query(..., description="用户提出的问题", min_length=1, max_length=1000),
    top_k: int = Query(default=3, description="返回结果数量", ge=1, le=10),
    filter_json: Optional[str] = Query(default=None, description="可选的元数据过滤条件（JSON 格式）")
):
    """
    知识库问答检索接口
    
    - **question**: 用户提出的问题
    - **top_k**: 返回的结果数量（默认 3，最大 10）
    - **filter_json**: 可选的元数据过滤条件，JSON 格式
    
    返回 TOP-3 相关信息及其信任值
    """
    if _global_client is None or _global_embeddings is None:
        raise HTTPException(status_code=503, detail="服务未完全初始化，请稍后重试")
    
    try:
        # 将查询向量化
        query_vector = _global_embeddings.embed_query(question)
        
        # 构建过滤条件（如果有）
        qdrant_filter = None
        if filter_json:
            import json
            try:
                filter_dict = json.loads(filter_json)
                qdrant_filter = _build_qdrant_filter(filter_dict)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"filter_json 格式错误: {str(e)}")
        
        # 执行搜索
        search_result = _global_client.query_points(
            collection_name=_config.COLLECTION_NAME,
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
            
            # 提取内容
            content = payload.get('page_content', '') if payload else ''
            doc_name = payload.get('doc_name') if payload else None
            headers = payload.get('headers') if payload else None
            
            # 计算信任值
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
        print(f"\n检索过程发生错误！详细堆栈信息如下：")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.get("/health", summary="健康检查接口")
async def health_check():
    """健康检查接口"""
    status = {
        "status": "healthy",
        "client_ready": _global_client is not None,
        "embeddings_ready": _global_embeddings is not None
    }
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
