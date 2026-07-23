import os
# os.environ["SAFETENSORS_FAST_GPU"] = "1"
import glob
import json
import hashlib
import sys
# import faulthandler
# faulthandler.enable(file=sys.stderr, all_threads=True)
# print("🟢 [1/4] 基础环境正常，开始测试 torch...")
# sys.stdout.flush()
# try:
#     import torch
#     print(f"🟢 [2/4] torch 导入成功！CUDA 可用: {torch.cuda.is_available()}")
# except Exception as e:
#     print(f"🔴 torch 报错: {e}")

# sys.stdout.flush()
# print("🟢 [3/4] 开始测试 sentence_transformers (HuggingFace底层)...")
# try:
#     import sentence_transformers
#     print("🟢 sentence_transformers 导入成功！")
# except Exception as e:
#     print(f"🔴 sentence_transformers 报错: {e}")

# sys.stdout.flush()
# print("🟢 [4/4] 开始测试 chromadb...")
# try:
#     import chromadb
#     print("🟢 chromadb 导入成功！")
# except Exception as e:
#     print(f"🔴 chromadb 报错: {e}")

# print("🎉 所有底层库导入测试通过！")

# sys.exit(1)

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import traceback


# ================= 1. 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
# 您下载的本地 bge-m3 模型文件夹路径
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model\\bge-m3-model") 
# 阶段三切片后落盘的 JSON 文件路径 (请替换为您实际的文件名)
CHUNKS_JSON_DIR = os.path.join(BASE_DIR, "data\\middle\\chunk_output") 
# Chroma 数据库本地持久化存储目录
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "mineru_rag_collection"
print(f"当前脚本所在绝对目录: {BASE_DIR}")
print(f"模型路径: {LOCAL_MODEL_PATH}")
print(f"数据路径: {CHUNKS_JSON_DIR}")
# sys.stdout.flush() # 强制刷新

# ================= 2. 数据加载与 Metadata 清洗 =================
def load_and_prepare_docs(json_path: str):
    """
    从 JSON 加载切片数据，并清洗 Metadata 以适配 Chroma 的严格类型限制
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到切片文件: {json_path}，请先运行 文本切块 生成数据。")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    docs = []
    for item in raw_data:
        clean_meta = {}
        # 【核心避坑】：Chroma 不支持 list 或 dict 类型的 metadata，必须序列化为字符串
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

# ================= 3. 初始化本地 BGE-M3 模型 =================
def init_local_bge_m3(model_path: str):
    """
    加载本地路径下的 BGE-M3 模型
    """
    print(f"正在加载本地 BGE-M3 模型: {model_path} ...")
    
    # 【诊断补丁 3】：自动检测 GPU，防止写死 cuda 导致底层崩溃
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"检测到 PyTorch 可用设备: {device}")
    
    model_kwargs = {
        'device': device,             # 如果有 NVIDIA GPU 且安装了 CUDA 版 PyTorch，请改为 'cuda'
        'trust_remote_code': True    # BGE-M3 必须开启此参数以加载自定义代码
    }
    encode_kwargs = {
        'normalize_embeddings': True # BGE 模型强烈推荐开启归一化，以使用余弦相似度
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

# ================= 4. 向量化并存入 Chroma =================
def embed_and_store(docs: list, embeddings: HuggingFaceEmbeddings):
    """
    将 Document 列表向量化并持久化到本地 Chroma 数据库
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    print(f"开始向量化 {len(docs)} 个 Chunks 并存入 {CHROMA_PERSIST_DIR} ...")
    
    # 如果数据库已存在，则加载；如果不存在，则创建
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"} # 使用余弦相似度
    )
    
    print(f"向量化完成！数据已持久化至本地目录: {CHROMA_PERSIST_DIR}")
    return vectorstore

# ================= 5. 检索测试 =================
def test_retrieval(vectorstore: Chroma, query: str, filter_dict: dict = None):
    """
    测试向量检索与标量过滤
    """
    print(f"\n正在检索: '{query}'")
    if filter_dict:
        print(f"启用 Metadata 过滤: {filter_dict}")
        
    results = vectorstore.similarity_search_with_relevance_scores(
        query, 
        k=3, 
        filter=filter_dict
    )
    
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Top {i+1} (相似度: {score:.4f}) ---")
        print(f"来源: {doc.metadata.get('doc_name')} | 章节: {doc.metadata.get('headers')}")
        print(f"内容: {doc.page_content[:100]}...")
        
# ================= 批量处理与测试运行模块 =================
def process_directory(input_dir: str, model_path: str, persist_dir: str, collection_name: str, file_pattern: str = "*_chunk.json"):
    """
    批量处理目录下的所有匹配文件，并持久化到同一个 Chroma 集合中
    """
    search_pattern = os.path.join(input_dir, "**", file_pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        print(f"未在 {input_dir} 下找到任何匹配 {file_pattern} 的文件。")
        return

    print(f"找到 {len(json_files)} 个文档，开始处理...\n")
    
    # 1. 初始化本地模型
    bge_m3_embedder = init_local_bge_m3(model_path)
    
    # 2. 初始化/连接 Chroma 向量库 (全局唯一实例)
    print(f"🔗 连接/创建 Chroma 数据库: {persist_dir} ...")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=bge_m3_embedder,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
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
                
                # B. 提取全局唯一 ID (优先使用阶段二生成的 block_id，保证幂等性)
                ids = []
                for doc in docs:
                    bid = doc.metadata.get("block_id")
                    if not bid:
                        # 兜底方案：如果没有 block_id，用文件名+内容MD5生成唯一ID
                        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]
                        bid = f"{os.path.basename(f_path)}_{content_hash}"
                    ids.append(bid)
                
                # C. 分批入库 (Batch Upsert)，防止大文件导致 OOM 或超时
                batch_size = 100
                for i in range(0, len(docs), batch_size):
                    batch_docs = docs[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    
                    texts = [d.page_content for d in batch_docs]
                    metadatas = [d.metadata for d in batch_docs]
                    
                    print(f"  ⏳ 正在使用 BGE-M3 计算 {len(texts)} 条向量...")
                    # 【核心修复】：手动调用 BGE-M3 模型计算向量！
                    batch_embeddings = bge_m3_embedder.embed_documents(texts)
                    # 将计算好的 embeddings 一并传入 upsert
                    # 直接调用底层的 upsert (存在则更新，不存在则插入)
                    vectorstore._collection.upsert(
                        embeddings=batch_embeddings, # 必须传入这个参数！
                        documents=texts,
                        metadatas=metadatas,
                        ids=batch_ids
                    )
                    total_upserted += len(batch_docs)
                    
                print(f"入库成功 (本文件 {len(docs)} 条 Chunk)")
                    
            except Exception as e:
                print(f"处理文件 {f_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"文件不存在: {f_path}")
            
    print(f"\n批量处理完成！共向量化并 Upsert {total_upserted} 条数据至集合 '{collection_name}'。")
        
        # 返回 vectorstore 实例，供后续检索测试使用
    return vectorstore

# ================= 6. 主执行流程 =================
if __name__ == "__main__":
    try:
        print("="*50)
        print("开始向量化存储流水线")
        print("="*50)
        print("开始向量化存储")
        # 1. 执行批量处理与入库，并获取数据库实例
        db = process_directory(
            input_dir=CHUNKS_JSON_DIR, 
            model_path=LOCAL_MODEL_PATH,
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            file_pattern="*_chunk.json" # 注意：如果您阶段三没有另存为 _chunk.json，这里应该匹配阶段二的产物 *_stage2.json 或在内存中直接切片
        )
        
        # 2. 模拟业务场景进行检索测试 (确保 db 不为空)
        if db:
            print("\n" + "="*50)
            print("开始检索测试")
            print("="*50)
            
            # 场景 A：纯语义检索
            test_retrieval(db, "1553B发生过什么问题")
            
            # 场景 B：结合 Metadata 的混合过滤检索
            # 假设只在“故障案例”中搜索“操作类”问题的处理措施
            test_retrieval(
                db, 
                "处理措施有哪些？", 
                filter_dict={
                    "$and": [
                        {"doc_type": {"$eq": "fault_case"}},
                        {"biz_问题分类": {"$eq": "操作"}}
                    ]
                }
            )
        else:
            print("数据库未初始化，跳过检索测试。")
    except Exception as e:
        print("\n❌ 发生严重错误！详细堆栈信息如下：")
        traceback.print_exc()