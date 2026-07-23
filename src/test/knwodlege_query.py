import os
import glob

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
# Qdrant 相关导入
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
import traceback


# ================= 1. 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
# 您下载的本地 bge-m3 模型文件夹路径
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model", "bge-m3-model") 

# 【Qdrant 本地模式配置】无需 Docker，无需网络
QDRANT_LOCAL_PATH = os.path.join(BASE_DIR, "qdrant_local")
COLLECTION_NAME = "mineru_rag_collection"
# 向量维度，BGE-M3 默认为 1024
VECTOR_SIZE = 1024

print(f"当前脚本所在绝对目录: {BASE_DIR}")
print(f"模型路径: {LOCAL_MODEL_PATH}")
print(f"Qdrant 本地存储路径: {QDRANT_LOCAL_PATH}")

# ================= 2. 初始化 Qdrant 客户端（本地文件模式） =================
def init_qdrant_client():
    """
    初始化 Qdrant 客户端（本地文件模式，无需 Docker，无需网络）
    """
    os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_LOCAL_PATH)
    print(f"使用 Qdrant 本地文件模式，数据存储于: {QDRANT_LOCAL_PATH}")
    return client

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024):
    """
    检查并创建 Collection（如果不存在）
    """
    try:
        # 检查集合是否存在
        client.get_collection(collection_name=collection_name)
        print(f"集合 '{collection_name}' 已存在，复用。")
    except Exception:
        # 集合不存在，创建
        print(f"集合 '{collection_name}' 不存在，正在创建...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE  # 使用余弦相似度
            )
        )
        print(f"集合 '{collection_name}' 创建成功（维度: {vector_size}, 距离: Cosine）")

# ================= 4. 初始化本地 BGE-M3 模型 =================
def init_local_bge_m3(model_path: str):
    """
    加载本地路径下的 BGE-M3 模型
    """
    print(f"正在加载本地 BGE-M3 模型: {model_path} ...")

    import torch
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


def _build_qdrant_filter(filter_dict: dict):
    """
    将 Chroma 风格的 filter_dict 转换为 Qdrant 的 Filter 对象

    支持简单转换：
    - {"key": {"$eq": "value"}} -> FieldCondition(key="key", match=MatchValue(value="value"))
    - {"$and": [...]} -> Filter(must=[...])
    """
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

# ================= 6. 检索测试 =================
def test_retrieval_qdrant(
    client: QdrantClient, 
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    query: str, 
    filter_dict: dict = None
):
    """
    测试向量检索与标量过滤（Qdrant 版本）
    """
    print(f"\n正在检索: '{query}'")

    # 将查询向量化
    query_vector = embeddings.embed_query(query)

    # 构建 Qdrant 过滤条件
    qdrant_filter = None
    if filter_dict:
        print(f"启用 Metadata 过滤: {filter_dict}")
        qdrant_filter = _build_qdrant_filter(filter_dict)

    # 执行搜索 - 使用 client.query_points() 替代 client.search()
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=3,
        with_payload=True,
        with_vectors=False
    )

    for i, point in enumerate(search_result.points):
        score = point.score
        payload = point.payload
        print(f"\n--- Top {i+1} (相似度: {score:.4f}) ---")
        print(f"来源: {payload.get('doc_name')} | 章节: {payload.get('headers')}")
        print(f"内容: {payload.get('page_content', '')[:100]}...")

if __name__ == "__main__":
    client = None
    try:
        print("="*50)
        print("开始本地知识检索")
        print("="*50)

        # 1. 初始化 Qdrant 客户端
        client = init_qdrant_client()

        # 2. 初始化本地模型
        bge_m3_embedder = init_local_bge_m3(LOCAL_MODEL_PATH)

        # 场景 A：纯语义检索
        test_retrieval_qdrant(
            client, bge_m3_embedder, COLLECTION_NAME,
            "电池激活输出异常故障发生在什么时间"
        )

        # 场景 B：结合 Metadata 的混合过滤检索
        # test_retrieval_qdrant(
        #     client, bge_m3_embedder, COLLECTION_NAME,
        #     "电池激活输出异常故障发生在什么时间", 
        #     filter_dict={
        #         "$and": [
        #             {"doc_type": {"$eq": "fault"}},
        #             {"问题分类": {"$eq": "设计"}}
        #         ]
        #     }
        # )
    except Exception as e:
        print("\n发生严重错误！详细堆栈信息如下：")
        traceback.print_exc()
    finally:
        # 确保 client 被关闭，避免 __del__ 异常
        if client is not None:
            try:
                client.close()
            except Exception:
                pass



