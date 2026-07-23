# export_to_html.py
import chromadb
import json

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("mineru_rag_collection")  # 修改为你的集合名
data = collection.get()

html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ChromaDB 数据查看器</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .card { background: white; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .id { color: #666; font-size: 12px; }
        .content { margin: 10px 0; line-height: 1.6; }
        .meta { background: #f0f0f0; padding: 8px; border-radius: 4px; font-size: 13px; }
        h1 { color: #333; }
        .stats { background: #e3f2fd; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>📦 ChromaDB 数据查看器</h1>
    <div class="stats">
        <strong>集合:</strong> mineru_rag_collection | 
        <strong>总记录数:</strong> """ + str(len(data['ids'])) + """
    </div>
"""

for i, doc_id in enumerate(data['ids']):
    meta = json.dumps(data['metadatas'][i], ensure_ascii=False) if data['metadatas'][i] else "{}"
    content = data['documents'][i] if data['documents'][i] else ""
    # 转义 HTML
    content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"""
    <div class="card">
        <div class="id">ID: {doc_id}</div>
        <div class="content">{content}</div>
        <div class="meta">📋 Metadata: {meta}</div>
    </div>
    """

html += "</body></html>"

with open("chroma_viewer.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ 已生成 chroma_viewer.html，用浏览器打开即可查看")