import re
import markdownify
from typing import List, Dict
from baseParseStrategy import BaseParseStrategy, UnifiedMetadata
from bs4 import BeautifulSoup

class DeclarationStrategy(BaseParseStrategy):
    def __init__(self):
        self.level_patterns = [
            (r'^[一二三四五六七八九十]+、', 1),  
            (r'^[（\(][一二三四五六七八九十]+[）\)]', 2), 
            (r'^\d+\.\s*', 3) 
        ]

    def _extract_biz_info_from_table(self, html: str) -> dict:
        """从申报书首页表格中提取核心业务属性"""
        biz_info = {}
        if not html:
            return biz_info
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # 定义我们关心的核心业务字段
            target_keys = ["项目名称", "承担单位", "项目申请单位", "项目负责人", "所属领域方向", "技术领域", "项目起止时间"]
            
            for row in soup.find_all('tr'):
                cols = row.find_all(['td', 'th'])
                row_texts = [c.get_text(strip=True) for c in cols]
                for i, text in enumerate(row_texts):
                    for key in target_keys:
                        # 如果当前列包含目标 Key，且后面还有列，则提取后面的列作为 Value
                        if key in text and i + 1 < len(row_texts):
                            val = row_texts[i+1]
                            # 过滤掉空值或重复的 key
                            if val and val != key and key not in biz_info:
                                biz_info[key] = val
        except Exception:
            pass
        return biz_info

    # ... (_extract_content 和 _update_headers_stack 保持您之前的代码不变) ...
    def _extract_content(self, b: Dict) -> Dict:
        b_type = b.get("type", b.get("type ", "")).strip()
        page = b.get("page_idx", b.get("page_idx ", 0))
        text, html = "", ""
        if "text" in b or "text " in b:
            text = b.get("text", b.get("text ", "")).strip()
        elif "content" in b or "content " in b:
            content_obj = b.get("content", b.get("content ", {}))
            if isinstance(content_obj, dict):
                p_list = content_obj.get("paragraph_content", content_obj.get("paragraph_content ", []))
                if p_list: text = "".join([item.get("content", item.get("content ", "")) for item in p_list if isinstance(item, dict)]).strip()
                t_list = content_obj.get("title_content", content_obj.get("title_content ", []))
                if t_list and not text: text = "".join([item.get("content", item.get("content ", "")) for item in t_list if isinstance(item, dict)]).strip()
                html = content_obj.get("html", content_obj.get("html ", "")).strip()
        return {"type": b_type, "page": page, "text": text, "html": html}

    def _update_headers_stack(self, text: str, headers_stack: List[str]) -> List[str]:
        current_level = 3 
        for pattern, lvl in self.level_patterns:
            if re.match(pattern, text):
                current_level = lvl
                break
        new_stack = headers_stack[:current_level-1]
        new_stack.append(text.replace("：", "").replace(":", "").strip())
        return new_stack

    def parse(self, blocks: List[Dict], file_name: str, doc_type: str) -> List[Dict]:
        raw_docs = []
        headers_stack = []
        biz_info = {} # 初始化业务信息字典
        biz_info_extracted = False # 标记是否已经提取过
        
        clean_doc_name = file_name.split('_')[0] if '_' in file_name else file_name
        
        for b in blocks:
            extracted = self._extract_content(b)
            b_type, page, text, html = extracted["type"], extracted["page"], extracted["text"], extracted["html"]
            
            if b_type == "page_footer" or not (text or html):
                continue

            # 【核心提取】：遇到第一个表格时，提取申报书的 biz_info
            if not biz_info_extracted and b_type == "table" and html:
                biz_info = self._extract_biz_info_from_table(html)
                biz_info_extracted = True

            if b_type == "title" and text:
                headers_stack = self._update_headers_stack(text, headers_stack)
                continue
                
            page_num = int(page) + 1 if isinstance(page, (int, float)) else 1
            base_metadata = {
                "source": file_name,
                "doc_name": clean_doc_name,
                "doc_type": doc_type,
                "page_range": [page_num, page_num],
                "headers": " > ".join(headers_stack)
            }

            if b_type == "table" and html:
                md_table = markdownify.markdownify(html, heading_style="ATX").strip()
                meta = UnifiedMetadata(**base_metadata, chunk_type="table")
                meta.biz_info = biz_info.copy() # 挂载 biz_info
                raw_docs.append({"content": md_table, "metadata": meta.to_dict()})
                
            elif text:
                meta = UnifiedMetadata(**base_metadata, chunk_type="text")
                meta.biz_info = biz_info.copy() # 挂载 biz_info
                raw_docs.append({"content": text, "metadata": meta.to_dict()})
                
        return raw_docs