# view_chroma.py

import chromadb
import json

client = chromadb.PersistentClient(path="./chroma_db")

print("=" * 70)
print("📦 ChromaDB 数据查看器（含向量展示）")
print("=" * 70)

collections = client.list_collections()
print(f"\n🔍 发现 {len(collections)} 个集合:")
for c in collections:
    print(f"   - {c.name}")

if not collections:
    print("   (无集合，数据库可能为空)")
    exit()

for collection in collections:
    print(f"\n{'='*70}")
    print(f"📂 集合名称: {collection.name}")
    print(f"{'='*70}")
    

    count = collection.count()
    print(f"   总记录数: {count}")
    
    if count == 0:
        print("   (空集合)")
        continue
    
    # 获取数据，显式要求返回 embeddings
    raw_data = collection.get(include=["embeddings", "documents", "metadatas"], limit=5)
    
    # 【关键修复】将 numpy ndarray 全部转为 Python list，彻底规避布尔判断问题
    ids = raw_data.get('ids', [])
    docs = raw_data.get('documents', []) or []
    metas = raw_data.get('metadatas', []) or []
    embeddings_raw = raw_data.get('embeddings', [])
    
    # 统一转为 Python list（兼容 numpy ndarray 和原生 list）
    if embeddings_raw is not None:
        if hasattr(embeddings_raw, 'tolist'):
            embeddings = embeddings_raw.tolist()
        else:
            embeddings = list(embeddings_raw)
    else:
        embeddings = []
    
    # 判断是否有向量数据：用 len()，绝不用 if embeddings:
    has_embeddings = len(embeddings) > 0 and len(embeddings[0]) > 0
    vector_dim = len(embeddings[0]) if has_embeddings else 0
    print(f"   向量维度: {vector_dim}")
    
    print(f"\n   👁 前 {min(len(ids), 5)} 条数据预览:")
    
    for i in range(len(ids)):
        doc_id = ids[i]
        doc = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        emb = embeddings[i] if i < len(embeddings) else []
        
        print(f"\n   --- 记录 {i+1} (ID: {doc_id}) ---")
        
        # 文本内容（截断）
        content = doc if len(doc) < 200 else doc[:200] + "..."
        print(f"   内容: {content}")
        
        # 元数据
        print(f"   元数据: {json.dumps(meta, ensure_ascii=False) if meta else '{}'}")
        
        # 【向量展示】emb 已经是 Python list，安全操作
        if len(emb) > 0:
            print(f"   向量维度: {len(emb)}")
            print(f"   向量前 5 个值: {emb[:5]}")
            print(f"   向量后 5 个值: {emb[-5:]}")
        else:
            print("   向量: 未获取")
    
    # 导出选项
    export = input(f"\n💾 是否导出集合 '{collection.name}' 全部数据到 JSON? (y/n): ").strip().lower()
    if export == 'y':
        all_raw = collection.get(include=["embeddings", "documents", "metadatas"])
        
        # 同样处理：numpy -> list
        all_data = {
            'ids': all_raw.get('ids', []),
            'documents': all_raw.get('documents', []) or [],
            'metadatas': all_raw.get('metadatas', []) or [],
            'embeddings': []
        }
        
        all_emb_raw = all_raw.get('embeddings', [])
        if all_emb_raw is not None:
            if hasattr(all_emb_raw, 'tolist'):
                all_data['embeddings'] = all_emb_raw.tolist()
            else:
                all_data['embeddings'] = [list(e) if hasattr(e, 'tolist') else e for e in all_emb_raw]
        
        output_file = f"chroma_export_{collection.name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"   ✅ 已导出到: {output_file}")

print("\n" + "=" * 70)
print("💡 说明：向量是机器理解的'语义指纹'，文本是人类可读的内容。")
print("   检索时输入文本 → 转为向量 → 找相似向量 → 返回对应文本")
print("=" * 70)