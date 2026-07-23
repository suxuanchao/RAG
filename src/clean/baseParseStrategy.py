from typing import List, Dict
import re

# ================= 1. 统一元数据模型 (Unified Schema) =================
class UnifiedMetadata:
    """定义所有文档必须遵守的核心元数据标准"""
    def __init__(self, source: str, doc_name: str, doc_type: str, page_range: list, headers: str, chunk_type: str):
        self.source = source
        self.doc_name = doc_name
        self.doc_type = doc_type      
        self.page_range = page_range
        self.headers = headers
        self.chunk_type = chunk_type
        self.biz_info = {}            # 业务扩展属性字典

    def to_dict(self):
        meta_dict = {
            "source": self.source,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "page_range": self.page_range,
            "headers": self.headers,
            "chunk_type": self.chunk_type
        }
        # 【核心修改】：将 biz_info 打平，加上 biz_ 前缀，防止与核心字段冲突
        for k, v in self.biz_info.items():
            # 清理 key 中的空格，确保符合数据库列名规范
            clean_k = re.sub(r'\s+', '_', k.strip())
            meta_dict[f"{clean_k}"] = v
        return meta_dict

# ================= 2. 策略基类 (Base Strategy) =================
class BaseParseStrategy:
    def parse(self, blocks: List[Dict], file_path: str, doc_type: str) -> List[Dict]:
        raise NotImplementedError
