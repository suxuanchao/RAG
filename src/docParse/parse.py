#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 MinerU API 的文档解析模块

功能：
1. 调用 MinerU API 解析 PDF/Word 等文档
2. 将解析结果输出到 /data/middle/mineru_output 目录
3. 支持批量处理多个文档
4. 支持断点续传（跳过已解析的文件）

使用方式：
    python parse.py --input_dir /path/to/input/pdfs --output_dir /data/middle/mineru_output
    python parse.py --file /path/to/single.pdf
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Windows 控制台默认 GBK 编码，强制 stdout/stderr 使用 UTF-8，
# 否则输出 ✓/⚠ 等 Unicode 符号会抛 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 默认输出目录（跨平台：放在用户主目录下，避免 Linux 风格路径在 Windows 上失效）
DEFAULT_OUTPUT_DIR = str(Path.home() / "mineru_output")


def get_file_md5(file_path: str) -> str:
    """计算文件的 MD5 值，用于判断文件是否变化"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def check_mineru_available() -> bool:
    """检查 MinerU 是否可用"""
    try:
        # 尝试导入 magic-pdf (MinerU 的核心包)
        import magic_pdf
        print(f"✓ MinerU (magic-pdf) 已安装，版本：{magic_pdf.__version__ if hasattr(magic_pdf, '__version__') else 'unknown'}")
        return True
    except ImportError:
        print("✗ 未找到 MinerU (magic-pdf)，请安装:")
        print("  pip install magic-pdf")
        print("  或参考官方文档：https://github.com/opendatalab/MinerU")
        return False


def parse_with_mineru_api(
    input_file: str,
    output_dir: str,
    force: bool = False,
    lang: Optional[str] = None,
    formula_enable: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    使用 MinerU API 解析单个文档

    Args:
        input_file: 输入文件路径（PDF/Word 等）
        output_dir: 输出目录
        force: 是否强制重新解析（即使已存在输出）
        lang: OCR 语言（如 "ch"、"en"），None 表示自动检测
        formula_enable: 是否启用公式识别（UniMERNet）。
            当前环境 transformers 版本与 UniMERNet 不兼容，默认关闭；
            若已安装兼容版本可通过 --formula 开启。

    Returns:
        解析结果字典，包含 status、output_file 等信息
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"✗ 文件不存在：{input_file}")
        return {"status": "error", "message": f"File not found: {input_file}"}

    # 生成输出文件名
    file_name = input_path.stem
    output_json_path = Path(output_dir) / f"{file_name}.json"

    # 检查是否已解析过（通过 MD5 判断文件是否变化）
    if output_json_path.exists() and not force:
        # 读取已存在的解析结果
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            # 如果已有解析结果，检查源文件是否变化
            if existing_data.get('metadata', {}).get('source_md5') == get_file_md5(input_file):
                print(f"⊘ 跳过（已解析且源文件未变化）: {input_file}")
                print(f"  输出文件：{output_json_path}")
                return {
                    "status": "skipped",
                    "output_file": str(output_json_path),
                    "message": "File already parsed and source unchanged"
                }
        except Exception as e:
            print(f"⚠ 读取已有解析结果失败：{e}，将重新解析")

    print(f"→ 开始解析：{input_file}")

    try:
        # 导入 MinerU 新版 API（magic-pdf 1.x / MinerU 2.x）
        # 旧版 magic_pdf.pipe.UNIPipe 已被移除，新版统一使用
        # PymuDocDataset + operators 流水线，OCR/文本模式由框架自动调度
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
        from magic_pdf.config.constants import MODEL_NAME

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取 PDF 原始字节
        with open(input_file, "rb") as f:
            pdf_bytes = f.read()
        if not pdf_bytes:
            raise ValueError(f"PDF 文件为空：{input_file}")

        # 构建 Dataset
        ds = PymuDocDataset(pdf_bytes, lang=lang)

        # 自动判断解析方式（等价于官方 CLI 的 method='auto'）
        pdf_type = ds.classify()
        use_ocr = (pdf_type != SupportedPdfParseMethod.TXT)
        print(f"  PDF 类型：{pdf_type}（{'OCR 模式' if use_ocr else '文本模式'}）")

        # 准备图片输出目录（OCR/含图 PDF 会导出图片）
        image_dir_name = f"{file_name}_images"
        local_image_dir = Path(output_dir) / image_dir_name
        os.makedirs(local_image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(str(local_image_dir))

        # 推理 + 解析流水线
        # 显式指定 layout_model=doclayout_yolo：默认 layoutlmv3 依赖 detectron2，
        # 在 Windows 上极难安装；doclayout_yolo 已随模型包下载且无此依赖。
        infer_result = ds.apply(
            doc_analyze,
            ocr=use_ocr,
            lang=lang,
            layout_model=MODEL_NAME.DocLayout_YOLO,
            formula_enable=formula_enable,
        )
        if use_ocr:
            pipe_result = infer_result.pipe_ocr_mode(image_writer, debug_mode=False, lang=lang)
        else:
            pipe_result = infer_result.pipe_txt_mode(image_writer, debug_mode=False, lang=lang)

        # 直接获取解析结果（无需落盘中间文件）
        # - markdown: 完整 markdown 字符串
        # - content_list: 结构化分页内容列表
        md_content = pipe_result.get_markdown(image_dir_name) or ""
        content_list = pipe_result.get_content_list(image_dir_name)
        if isinstance(content_list, str):
            content_list = json.loads(content_list)
        if not isinstance(content_list, list):
            content_list = []

        # 真正无内容时按失败处理，避免空结果被缓存为"成功"而永久跳过
        if not md_content.strip() and not content_list:
            raise ValueError(
                "未能从 PDF 中提取到任何文本内容。"
                "可能原因：1) PDF 为扫描版且 OCR 模型权重未就绪；"
                "2) PDF 加密或损坏。请检查 MinerU 模型是否已下载。"
            )

        # 构建输出 JSON（兼容后续 clean 模块的格式）
        output_data = {
            "metadata": {
                "source_file": str(input_path.absolute()),
                "source_md5": get_file_md5(input_file),
                "parse_time": datetime.now().isoformat(),
                "parser": "mineru",
                "file_name": file_name,
                "file_size": input_path.stat().st_size,
                "parse_method": "ocr" if use_ocr else "txt",
                "lang": lang
            },
            "markdown": md_content,
            "content": []
        }

        # 将 content_list 结构化为统一 schema：{page_num, type, content}
        for item in content_list:
            if not isinstance(item, dict):
                continue
            page_idx = item.get("page_idx", 0)
            item_type = item.get("type", "text")
            if item_type == "image":
                # 图片：记录相对路径，附带 caption/footnote（如有）
                parts = [item.get("img_path", "")]
                for key in ("img_caption", "img_footnote"):
                    cap = item.get(key)
                    if cap:
                        parts.append(" ".join(cap) if isinstance(cap, list) else str(cap))
                content = "\n".join(p for p in parts if p)
            elif item_type == "table":
                content = item.get("table_body") or item.get("text") or ""
                cap = item.get("table_caption") or item.get("table_caption_list")
                if cap:
                    cap_str = " ".join(cap) if isinstance(cap, list) else str(cap)
                    content = f"{cap_str}\n{content}" if content else cap_str
            else:
                content = item.get("text", "") or ""

            if isinstance(content, str) and content.strip():
                output_data["content"].append({
                    "page_num": page_idx + 1,
                    "type": item_type,
                    "content": content
                })

        # 兜底：若 content_list 为空但 markdown 有内容，则按 markdown 整体存一条
        if not output_data["content"] and md_content.strip():
            output_data["content"].append({
                "page_num": 1,
                "type": "text",
                "content": md_content
            })

        # 保存解析结果
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 解析完成：{input_file}")
        print(f"  输出文件：{output_json_path}")
        print(f"  内容块数：{len(output_data['content'])}")

        return {
            "status": "success",
            "output_file": str(output_json_path),
            "pages": len(output_data["content"]),
            "message": "Parsing completed successfully"
        }

    except Exception as e:
        error_msg = f"解析失败：{str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()

        # 保存错误信息
        error_output = {
            "metadata": {
                "source_file": str(input_path.absolute()),
                "source_md5": get_file_md5(input_file),
                "parse_time": datetime.now().isoformat(),
                "parser": "mineru",
                "file_name": file_name,
                "error": error_msg
            },
            "content": []
        }

        error_json_path = Path(output_dir) / f"{file_name}_error.json"
        with open(error_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_output, f, ensure_ascii=False, indent=2)

        return {
            "status": "error",
            "output_file": str(error_json_path),
            "message": error_msg
        }


def batch_parse(
    input_dir: str,
    output_dir: str,
    extensions: List[str] = None,
    force: bool = False,
    lang: Optional[str] = None,
    formula_enable: bool = False,
) -> Dict[str, Any]:
    """
    批量解析目录中的所有文档

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        extensions: 支持的扩展名列表
        force: 是否强制重新解析
        lang: OCR 语言

    Returns:
        统计信息字典
    """
    if extensions is None:
        extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt']

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"✗ 输入目录不存在：{input_dir}")
        return {"total": 0, "success": 0, "skipped": 0, "error": 0}

    # 查找所有符合条件的文件
    # 注意：Windows 文件系统大小写不敏感，.pdf 与 .PDF 会匹配同一文件，
    # 因此用归一化路径去重，避免每个文件被处理两次。
    seen = set()
    files_to_parse = []
    for ext in extensions:
        for pattern in (f"**/*{ext}", f"**/*{ext.upper()}"):
            for p in input_path.glob(pattern):
                key = str(p.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    files_to_parse.append(p)

    if not files_to_parse:
        print(f"⚠ 在 {input_dir} 中未找到任何文档 (支持格式：{', '.join(extensions)})")
        return {"total": 0, "success": 0, "skipped": 0, "error": 0}

    print(f"\n{'='*60}")
    print(f"批量解析任务")
    print(f"{'='*60}")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"文件数量：{len(files_to_parse)}")
    print(f"支持格式：{', '.join(extensions)}")
    print(f"{'='*60}\n")

    # 统计信息
    stats = {
        "total": len(files_to_parse),
        "success": 0,
        "skipped": 0,
        "error": 0,
        "details": []
    }

    # 逐个解析
    for idx, file_path in enumerate(files_to_parse, 1):
        print(f"\n[{idx}/{len(files_to_parse)}] ", end="")
        result = parse_with_mineru_api(str(file_path), output_dir, force, lang=lang, formula_enable=formula_enable)

        if result["status"] == "success":
            stats["success"] += 1
        elif result["status"] == "skipped":
            stats["skipped"] += 1
        else:
            stats["error"] += 1

        stats["details"].append({
            "file": str(file_path),
            "result": result
        })

    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"解析完成统计")
    print(f"{'='*60}")
    print(f"总文件数：{stats['total']}")
    print(f"成功解析：{stats['success']}")
    print(f"跳过（未变化）：{stats['skipped']}")
    print(f"解析失败：{stats['error']}")
    print(f"{'='*60}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="基于 MinerU API 的文档解析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 解析单个文件
  python parse.py --file /path/to/document.pdf

  # 批量解析目录
  python parse.py --input_dir /path/to/pdfs --output_dir ./mineru_output

  # 强制重新解析所有文件
  python parse.py --input_dir /path/to/pdfs --force

  # 指定 OCR 语言（中文）
  python parse.py --file document.pdf --lang ch
        """
    )

    parser.add_argument("--file", "-f", type=str, help="单个文件路径")
    parser.add_argument("--input_dir", "-i", type=str, help="输入目录路径（批量处理）")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录路径（默认：{DEFAULT_OUTPUT_DIR}）")
    parser.add_argument("--force", action="store_true", help="强制重新解析所有文件（忽略缓存）")
    parser.add_argument("--extensions", "-e", type=str, nargs="+",
                        default=[".pdf"], help="支持的扩展名列表（默认：.pdf）")
    parser.add_argument("--lang", "-l", type=str, default=None,
                        help="OCR 语言（如 ch/en），默认自动检测")
    parser.add_argument("--formula", action="store_true",
                        help="启用公式识别（UniMERNet）。默认关闭：当前环境 transformers 版本与 UniMERNet 不兼容，"
                             "启用前需先安装兼容版本（如 transformers<4.40）")

    args = parser.parse_args()

    # 检查 MinerU 是否可用
    if not check_mineru_available():
        sys.exit(1)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 执行解析
    if args.file:
        # 单文件模式
        result = parse_with_mineru_api(args.file, args.output_dir, args.force,
                                       lang=args.lang, formula_enable=args.formula)
        if result["status"] == "error":
            sys.exit(1)
    elif args.input_dir:
        # 批量模式
        stats = batch_parse(args.input_dir, args.output_dir, args.extensions, args.force,
                            lang=args.lang, formula_enable=args.formula)
        if stats["total"] == 0:
            sys.exit(0)
        if stats["error"] > 0 and stats["success"] == 0:
            sys.exit(1)
    else:
        parser.print_help()
        print("\n✗ 错误：请指定 --file 或 --input_dir 参数")
        sys.exit(1)

    print("\n✓ 所有任务完成！")


if __name__ == "__main__":
    main()
