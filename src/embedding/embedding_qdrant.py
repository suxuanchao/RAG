import os
import glob
import json
import hashlib
import sys
import uuid
import atexit

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
# 阶段三切片后落盘的 JSON 文件路径
CHUNKS_JSON_DIR = os.path.join(BASE_DIR, "data", "middle", "chunk_output") 

# 【Qdrant 本地模式配置】无需 Docker，无需网络
QDRANT_LOCAL_PATH = os.path.join(BASE_DIR, "qdrant_local")
COLLECTION_NAME = "mineru_rag_collection"
# 向量维度，BGE-M3 默认为 1024
VECTOR_SIZE = 1024

print(f"当前脚本所在绝对目录: {BASE_DIR}")
print(f"模型路径: {LOCAL_MODEL_PATH}")
print(f"数据路径: {CHUNKS_JSON_DIR}")
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

# ================= 3. 数据加载与 Metadata 清洗 =================
def load_and_prepare_docs(json_path: str):
    """
    从 JSON 加载切片数据，并清洗 Metadata 以适配 Qdrant 的 payload 要求
    【注意】Qdrant 的 payload 支持嵌套 dict/list，比 Chroma 更灵活
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到切片文件: {json_path}，请先运行 文本切块 生成数据。")

    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    docs = []
    for item in raw_data:
        clean_meta = {}
        # Qdrant 的 payload 支持 list/dict 类型，无需强制序列化为字符串
        # 但为了统一性和兼容性，仍建议对复杂类型做处理
        for k, v in item['metadata'].items():
            if isinstance(v, (list, dict)):
                clean_meta[k] = json.dumps(v, ensure_ascii=False)
            elif v is None:
                clean_meta[k] = ""
            else:
                clean_meta[k] = v

        docs.append(Document(page_content=item['page_content'], metadata=clean_meta))

    print(f"成功加载 {len(docs)} 个 Document 对象，并完成 Metadata 类型清洗。")
    return docs

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

# ================= 5. 向量化并存入 Qdrant =================
def embed_and_store_to_qdrant(
    docs: list, 
    embeddings: HuggingFaceEmbeddings, 
    client: QdrantClient,
    collection_name: str
):
    """
    将 Document 列表向量化并持久化到 Qdrant
    【核心改动】：使用 Qdrant 原生 client.upsert 替代 Chroma.from_documents
    """
    print(f"开始向量化 {len(docs)} 个 Chunks 并存入 Qdrant 集合 '{collection_name}' ...")

    # 分批处理，防止 OOM
    batch_size = 100
    total_upserted = 0

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        texts = [d.page_content for d in batch_docs]
        metadatas = [d.metadata for d in batch_docs]

        # 生成唯一 ID（必须是整数或 UUID 格式）
        ids = []
        for idx, doc in enumerate(batch_docs):
            bid = doc.metadata.get("block_id")
            if bid and isinstance(bid, (int, str)):
                try:
                    ids.append(int(bid))
                except (ValueError, TypeError):
                    # 生成确定性 UUID（基于内容哈希）
                    content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                    ids.append(str(uuid.UUID(content_hash[:32])))
            else:
                # 兜底：生成确定性 UUID
                content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                ids.append(str(uuid.UUID(content_hash[:32])))

        print(f"  正在使用 BGE-M3 计算第 {i+1}-{min(i+batch_size, len(docs))} 条向量...")
        batch_embeddings = embeddings.embed_documents(texts)

        # 构建 PointStruct 列表
        points = []
        for idx, (vec, text, meta) in enumerate(zip(batch_embeddings, texts, metadatas)):
            payload = {
                "page_content": text,
                **meta  # 将 metadata 展开到 payload 中
            }
            points.append(PointStruct(id=ids[idx], vector=vec, payload=payload))

        # 执行 upsert（存在则更新，不存在则插入）
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True  # 等待写入完成
        )
        total_upserted += len(batch_docs)
        print(f"  已 upsert {len(batch_docs)} 条数据")

    print(f"向量化完成！共 upsert {total_upserted} 条数据至集合 '{collection_name}'")

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

# ================= 批量处理与测试运行模块 =================
def process_directory(
    input_dir: str, 
    model_path: str, 
    collection_name: str, 
    file_pattern: str = "*_chunk.json"
):
    """
    批量处理目录下的所有匹配文件，并持久化到 Qdrant 集合中
    """
    search_pattern = os.path.join(input_dir, "**", file_pattern)
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"未在 {input_dir} 下找到任何匹配 {file_pattern} 的文件。")
        return None

    print(f"找到 {len(json_files)} 个文档，开始处理...\n")

    # 1. 初始化 Qdrant 客户端
    client = init_qdrant_client()

    # 2. 初始化本地模型
    bge_m3_embedder = init_local_bge_m3(model_path)

    # 3. 确保集合存在
    # 先测试模型获取维度
    test_vec = bge_m3_embedder.embed_query("test")
    vector_size = len(test_vec)
    ensure_collection(client, collection_name, vector_size)

    total_upserted = 0

    for f_path in json_files:
        if os.path.exists(f_path):
            print(f"\n正在处理: {os.path.basename(f_path)}")
            try:
                # A. 加载并清洗当前文件的数据
                docs = load_and_prepare_docs(f_path)
                if not docs:
                    print("文件内容为空，跳过。")
                    continue

                # B. 分批入库
                batch_size = 100
                for i in range(0, len(docs), batch_size):
                    batch_docs = docs[i:i+batch_size]
                    texts = [d.page_content for d in batch_docs]
                    metadatas = [d.metadata for d in batch_docs]

                    # 生成唯一 ID（必须是整数或 UUID 格式）
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

                    print(f"  正在使用 BGE-M3 计算 {len(texts)} 条向量...")
                    batch_embeddings = bge_m3_embedder.embed_documents(texts)

                    # 构建 PointStruct
                    points = []
                    for idx, (vec, text, meta) in enumerate(zip(batch_embeddings, texts, metadatas)):
                        payload = {"page_content": text, **meta}
                        points.append(PointStruct(id=ids[idx], vector=vec, payload=payload))

                    # Upsert 到 Qdrant
                    client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True
                    )
                    total_upserted += len(batch_docs)

                print(f"入库成功 (本文件 {len(docs)} 条 Chunk)")

            except Exception as e:
                print(f"处理文件 {f_path} 时出错: {e}")
                traceback.print_exc()
        else:
            print(f"文件不存在: {f_path}")

    print(f"\n批量处理完成！共向量化并 Upsert {total_upserted} 条数据至集合 '{collection_name}'。")

    # 返回 client 和 embedder，供后续检索测试使用
    return client, bge_m3_embedder

def view_data(
    client: QdrantClient, 
    collection_name: str,
    limit: int
    ):
    # ========== 数据查看调试区 ==========
    print("\n" + "="*50)
    print("数据查看调试")
    print("="*50)

    # 查看集合统计
    info = client.get_collection(collection_name)
    print(f"\n集合: {collection_name}")
    print(f"数据条数: {info.points_count}")
    print(f"向量维度: {info.config.params.vectors.size}")

    # 遍历查看前 5 条数据
    print("\n--- 前 5 条数据 ---")
    result = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    for i, point in enumerate(result[0]):
        print(f"\n[{i+1}] ID: {point.id}")
        print(f"    doc_name: {point.payload.get('doc_name')}")
        print(f"    doc_type: {point.payload.get('doc_type')}")
        print(f"    问题分类: {point.payload.get('问题分类')}")
        print(f"    headers: {point.payload.get('headers')}")
        content = point.payload.get('page_content', '')
        print(f"    content: {content[:80]}...")

    # 按 metadata 过滤查看
    print("\n--- 过滤查看 (doc_type='fault_case') ---")
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    filtered = client.scroll(
        collection_name=collection_name,
        limit=limit,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_type", match=MatchValue(value="default"))]
        ),
        with_payload=True
    )
    for i, point in enumerate(filtered[0]):
        print(f"[{i+1}] ID: {point.id} | doc_name: {point.payload.get('doc_name')}")

# ================= 7. 主执行流程 =================
if __name__ == "__main__":
    client = None
    try:
        print("="*50)
        print("开始向量化存储流水线 (Qdrant 本地模式)")
        print("="*50)

        # 1. 执行批量处理与入库
        result = process_directory(
            input_dir=CHUNKS_JSON_DIR, 
            model_path=LOCAL_MODEL_PATH,
            collection_name=COLLECTION_NAME,
            file_pattern="*_chunk.json"
        )

        # 2. 模拟业务场景进行检索测试
        if result:
            client, embedder = result
            print("\n" + "="*50)
            print("开始检索测试")
            print("="*50)

            # 场景 A：纯语义检索
            test_retrieval_qdrant(
                client, embedder, COLLECTION_NAME,
                "1533B发生过哪些问题"
            )

            # 场景 B：结合 Metadata 的混合过滤检索
            test_retrieval_qdrant(
                client, embedder, COLLECTION_NAME,
                "处理措施有哪些？", 
                filter_dict={
                    "$and": [
                        {"doc_type": {"$eq": "fault_case"}},
                        {"问题分类": {"$eq": "操作"}}
                    ]
                }
            )

            # ========== 数据查看调试区 ==========
            view_data(client, COLLECTION_NAME, 5)

            # 正常关闭 client
            print("\n正在关闭 Qdrant 客户端...")
            client.close()
            client = None
            print("Qdrant 客户端已安全关闭。")
        else:
            print("数据库未初始化，跳过检索测试。")
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
            
    