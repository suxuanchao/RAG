from typing import List, Dict, Any
import glob
import json
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 智能切片模块 (Core Chunking) =================
def chunk_documents(raw_docs: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 50) -> List[Document]:
    """
    对阶段二产出的 raw_docs 进行 LangChain 智能切片
    """
    final_docs = []
    # 使用 RecursiveCharacterTextSplitter 基于特定格式分块, 保证按中文标点语义切分 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    )

    for item in raw_docs:
        metadata = item["metadata"]
        content = item["content"]
        chunk_type = metadata.get("chunk_type", "text")
        
        if chunk_type == "table":
            # 【核心规则 1】：表格绝对不被切碎，整体作为一个 Document
            final_docs.append(Document(page_content=content, metadata=metadata))
        else:
            # 【核心规则 2】：正文过长时进行切片，并复制 metadata 给每一个子块
            splits = text_splitter.split_text(content)
            for split in splits:
                final_docs.append(Document(page_content=split, metadata=metadata.copy()))

    return final_docs

# ================= 持久化落盘 =================
def save_chunk_output(docs: List[Document], output_dir: str, clean_doc_name: str):
    """将 LangChain Document 列表落盘为 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 构造输出路径
    output_path = os.path.join(output_dir, f"{clean_doc_name}_chunk.json")
    
    # 将 Document 对象转换为可序列化的字典
    serializable_docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        
    print(f" 切片数据已落盘至: {output_path}")

def load_chunks_from_disk(file_path: str) -> List[Document]:
    """从磁盘加载切片数据，还原为 LangChain Document 对象"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in data]

# ================= 批量处理与测试运行模块 =================
def process_directory(input_dir: str, output_dir: str, file_pattern: str = "*_clean.json"):
    """
    批量处理目录下的所有匹配文件
    """
    search_pattern = os.path.join(input_dir, "**", file_pattern)

    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        print(f"未在 {input_dir} 下找到任何匹配 {file_pattern} 的文件")
        return

    print(f"找到 {len(json_files)} 个文档,开始处理...\n")
    chunk_number = 0

    for f_path in  json_files:
        if os.path.exists(f_path):
            print(f"正在处理 {os.path.basename(f_path)}")

            try:
                with open(f_path, 'r', encoding='utf-8') as f:

                    raw_data = json.load(f)

                    docs = chunk_documents(raw_data)

                    # 4. 存储中间清洗结果
                    chunk_doc_name = os.path.basename(f_path).split('_')[0]

                    save_chunk_output(docs, output_dir, chunk_doc_name)

                    chunk_number += 1
                    print(f"共生成 {len(docs)} 个langchan document切片,成功切片 {chunk_number} 个文件 \n")

            except Exception as e:
                print(f"处理文件 {f_path} 时出错 {e}")

        else:
            print(f"文件不存在: {f_path}")


def main():
    """
    批量处理目录下的所有匹配文件
    """
    INPUT_DIR = "..\\..\\data\\middle\\cleaned_output"

    OUTPUT_DIR = "..\\..\\data\\middle\\chunk_output"

    process_directory(INPUT_DIR, OUTPUT_DIR)

if __name__ == '__main__':
    main()