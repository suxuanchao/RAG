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


def parse_with_mineru_api(input_file: str, output_dir: str, force: bool = False) -> Optional[Dict[str, Any]]:
    """
    使用 MinerU API 解析单个文档
    
    Args:
        input_file: 输入文件路径（PDF/Word 等）
        output_dir: 输出目录
        force: 是否强制重新解析（即使已存在输出）
    
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
        # 导入 MinerU
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.data_reader_writer import FileBasedDataReader
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取输入文件
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(input_file)
        
        # 创建临时目录用于 MinerU 中间文件
        temp_dir = Path(output_dir) / "temp" / file_name
        os.makedirs(temp_dir, exist_ok=True)
        
        # 创建 writer
        image_writer = DiskReaderWriter(str(temp_dir))
        md_content_writer = DiskReaderWriter(str(temp_dir))
        
        # 构建 JIPipe 需要的数据
        jso_useful_key = {
            "_pdf_type": "",  # 自动检测
            "model_list": [],
            "page_num": 0
        }
        
        # 执行解析
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        
        # 获取解析结果
        md_content = pipe.pipe_mk_markdown(image_writer, drop_mode="none")
        
        # 构建输出 JSON（兼容后续 clean 模块的格式）
        output_data = {
            "metadata": {
                "source_file": str(input_path.absolute()),
                "source_md5": get_file_md5(input_file),
                "parse_time": datetime.now().isoformat(),
                "parser": "mineru",
                "file_name": file_name,
                "file_size": input_path.stat().st_size
            },
            "content": []
        }
        
        # 解析 markdown 内容并结构化
        # MinerU 返回的是 markdown 字符串列表，每个元素是一页
        if isinstance(md_content, list):
            for page_idx, page_content in enumerate(md_content):
                if page_content.strip():
                    output_data["content"].append({
                        "page_num": page_idx + 1,
                        "type": "text",
                        "content": page_content
                    })
        elif isinstance(md_content, str):
            # 如果是单个字符串，按页分割
            pages = md_content.split("\f")  # form feed 字符常用于分页
            for page_idx, page_content in enumerate(pages):
                if page_content.strip():
                    output_data["content"].append({
                        "page_num": page_idx + 1,
                        "type": "text",
                        "content": page_content
                    })
        
        # 保存解析结果
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 解析完成：{input_file}")
        print(f"  输出文件：{output_json_path}")
        print(f"  页数：{len(output_data['content'])}")
        
        return {
            "status": "success",
            "output_file": str(output_json_path),
            "pages": len(output_data['content']),
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


def batch_parse(input_dir: str, output_dir: str, extensions: List[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    批量解析目录中的所有文档
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        extensions: 支持的扩展名列表
        force: 是否强制重新解析
    
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
    files_to_parse = []
    for ext in extensions:
        files_to_parse.extend(input_path.glob(f"**/*{ext}"))
        files_to_parse.extend(input_path.glob(f"**/*{ext.upper()}"))
    
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
        result = parse_with_mineru_api(str(file_path), output_dir, force)
        
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
  python parse.py --input_dir /path/to/pdfs --output_dir /data/middle/mineru_output
  
  # 强制重新解析所有文件
  python parse.py --input_dir /path/to/pdfs --force
  
  # 指定输出目录（默认：/data/middle/mineru_output）
  python parse.py --file document.pdf --output_dir /custom/output/dir
        """
    )
    
    parser.add_argument("--file", "-f", type=str, help="单个文件路径")
    parser.add_argument("--input_dir", "-i", type=str, help="输入目录路径（批量处理）")
    parser.add_argument("--output_dir", "-o", type=str, default="/data/middle/mineru_output",
                        help=f"输出目录路径（默认：/data/middle/mineru_output）")
    parser.add_argument("--force", action="store_true", help="强制重新解析所有文件（忽略缓存）")
    parser.add_argument("--extensions", "-e", type=str, nargs="+", 
                        default=[".pdf"], help="支持的扩展名列表（默认：.pdf）")
    
    args = parser.parse_args()
    
    # 检查 MinerU 是否可用
    if not check_mineru_available():
        sys.exit(1)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行解析
    if args.file:
        # 单文件模式
        result = parse_with_mineru_api(args.file, args.output_dir, args.force)
        if result["status"] == "error":
            sys.exit(1)
    elif args.input_dir:
        # 批量模式
        stats = batch_parse(args.input_dir, args.output_dir, args.extensions, args.force)
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
