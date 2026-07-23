from typing import List, Dict, Any
import glob
import json
import os
import hashlib
from faultCaseStrategy import FaultCaseStrategy
from declarationStrategy import DeclarationStrategy

# ================= 5. 智能路由与主控 Pipeline =================
class MinerU_RAG_Pipeline:
    def __init__(self):
        # 注册策略路由表
        self.strategies = {
            "fault_case": FaultCaseStrategy(),
            "declaration": DeclarationStrategy()
        }
        # self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        
    # ================= 1. 数据结构展平模块 =================
    def flatten_json_structure(self, raw_data: List[Any]) -> List[Dict[str, Any]]:
        """
        自动检测并展平 MinerU JSON 数据结构 (兼容 1D 和 2D 数组)
        解决由于 MinerU 版本不同导致的 AttributeError
        """
        all_blocks = []
        if not raw_data:
            return all_blocks
            
        if isinstance(raw_data[0], dict):
            # 情况 A: 一维数组 (如 交付问题公开.json)
            all_blocks = raw_data
        elif isinstance(raw_data[0], list):
            # 情况 B: 二维数组 (如 test_content_list_v2.json)
            for page_idx, page_blocks in enumerate(raw_data):
                for block in page_blocks:
                    # 如果内部没有 page_idx，手动注入外层的索引作为页码
                    if "page_idx" not in block and "page_idx " not in block:
                        block["page_idx"] = page_idx
                    all_blocks.append(block)
        else:
            raise ValueError("无法识别的 JSON 结构")
            
        return all_blocks

    def detect_doc_type(self, file_dir: str) -> str:
        """智能嗅探文档类型"""
        if "质量案例" in file_dir:
            return "fault_case"
        elif "报告" in file_dir:
            return "declaration"
        return "default" # 兜底策略
    
    # ================= 文件名清洗工具模块 =================
    def get_clean_source_name(self, json_path: str) -> str:
        """
        处理源文件名：如果包含 '_'，则分割并只保留第一部分。
        例如：'交付问题公开_1782463767.7736237_content_list.json' -> '交付问题公开'
            'test_content_list_v2.json' -> 'test'
        """
        base_name = os.path.basename(json_path)
        name_without_ext = os.path.splitext(base_name)[0]
        if '_' in name_without_ext:
            return name_without_ext.split('_')[0]
        return name_without_ext
    
    # ================= 持久化落盘 =================
    def save_clean_output(self, raw_docs: list, output_dir: str, clean_doc_name: str):
        """
        将阶段二(数据清洗与结构重组)的结构化中间层数据保存到本地 JSON 文件
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成带 hash 的 block_id，方便后续增量更新或去重
        for doc in raw_docs:
            content_hash = hashlib.md5(doc["content"].encode('utf-8')).hexdigest()[:8]
            doc["block_id"] = f"{clean_doc_name}_{content_hash}"
            
        # 构造输出路径
        output_path = os.path.join(output_dir, f"{clean_doc_name}_clean.json")
        
        # 落盘保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(raw_docs, f, ensure_ascii=False, indent=2)
            
        print(f"{clean_doc_name} 数据清洗与结构重组阶段产物已保存至: {output_path}")
        return output_path

    def process(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 1. 展平数据
        flat_blocks = self.flatten_json_structure(raw_data)
        
        # 2. 路由分发
        pathList = json_path.split('\\')    # .split('_')[0]
        
        file_dir = pathList[-2]
        
        file_name = pathList[-1]
        
        clean_doc_name = file_name.split('_')[0] if '_' in file_name else file_name
        
        doc_type = self.detect_doc_type(file_dir)
        
        strategy = self.strategies.get(doc_type, self.strategies["fault_case"])
        
        # 3. 执行特定策略解析
        raw_docs = strategy.parse(flat_blocks, file_name, doc_type)
        
        # 4. 存储中间清洗结果
        self.save_clean_output(raw_docs, OUTPUT_DIR, clean_doc_name)
    
# ================= 批量处理与测试运行模块 =================
def process_directory(pipeLine: MinerU_RAG_Pipeline, input_dir: str, output_dir: str, file_pattern: str = "*content_list*.json"):
    """
    批量处理目录下的所有匹配文件
    """
    search_pattern = os.path.join(input_dir, "**", "**", file_pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        print(f"未在 {input_dir} 下找到任何匹配 {file_pattern} 的文件。")
        return

    print(f"找到 {len(json_files)} 个文档，开始处理...\n")
    
    for f_path in json_files:
        if os.path.exists(f_path):
            print(f"\n正在处理: {os.path.basename(f_path)}")
            try:
                pipeLine.process(f_path)
                    
            except Exception as e:
                print(f"处理文件 {f_path} 时出错: {e}")
        else:
            print(f"文件不存在: {f_path}")
    
# ================= 测试运行 =================
if __name__ == "__main__":
    
    # INPUT_DIR = "..\\..\\data\\middle\\mineru_output"

    INPUT_DIR = "D:\\Tools\\MinerU_1.3.12\\MinerU\\output"

    OUTPUT_DIR = "..\\..\\data\\middle\\cleaned_output"
    
    pipeline = MinerU_RAG_Pipeline()
    
    process_directory(pipeline,INPUT_DIR,OUTPUT_DIR)
    