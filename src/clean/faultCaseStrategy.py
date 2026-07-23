from typing import List, Dict
import re
from baseParseStrategy import BaseParseStrategy, UnifiedMetadata

class FaultCaseStrategy(BaseParseStrategy):
    def parse(self, blocks: List[Dict], file_name: str, doc_type: str) -> List[Dict]:
        raw_docs = []
        headers_stack = []
        biz_info = {} # 初始化业务信息字典
        buffer = []
        buffer_pages = []
        clean_doc_name = file_name.split('_')[0] if '_' in file_name else file_name

        def save_buffer():
            if buffer:
                meta = UnifiedMetadata(
                    source=file_name, doc_name=clean_doc_name, doc_type=doc_type,
                    page_range=[min(buffer_pages)+1, max(buffer_pages)+1],
                    headers=" > ".join(headers_stack), chunk_type="text"
                )
                meta.biz_info = biz_info.copy() # 挂载 biz_info
                raw_docs.append({"content": "\n".join(buffer), "metadata": meta.to_dict()})
                buffer.clear(); buffer_pages.clear()

        for b in blocks:
            # 兼容带空格的 Key
            text = b.get("text", b.get("text ", "")).strip()
            page = b.get("page_idx", b.get("page_idx ", 0))
            if not text: continue

            # 【核心提取】：提取全局业务属性 (如 "问题分类：操作")
            # 条件：还没遇到章节标题，且包含冒号，且不是标准章节序号
            if not headers_stack and "：" in text and not re.match(r'^[一二三四五六七八九十]+、', text):
                k, v = text.split("：", 1)
                k = k.strip()
                v = v.strip()
                if v: 
                    biz_info[k] = v
                    continue
            
            # 兼容没有冒号的第一个 block (如 "典型质量问题案例3")
            if not headers_stack and not biz_info and "案例" in text and "：" not in text:
                biz_info["案例名称"] = text
                continue

            # 识别章节标题 (遇到标题则保存之前的 buffer)
            if re.match(r'^[一二三四五六七八九十]+、', text) or text.endswith("："):
                save_buffer()
                clean_title = text.replace("：", "").replace(":", "").strip()
                headers_stack = [clean_title]
            else:
                buffer.append(text)
                if isinstance(page, (int, float)):
                    buffer_pages.append(int(page))
        
        save_buffer()
        return raw_docs