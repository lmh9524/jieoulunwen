#!/usr/bin/env python3
"""
数据集下载脚本
用于下载VAW和CelebA数据集，包含断点续传和重试机制
"""

import os
import requests
import zipfile
import json
from urllib.parse import urlparse
import time

def download_file_with_resume(url, filepath, chunk_size=8192, max_retries=3):
    """
    带断点续传的文件下载功能
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # 如果文件已存在，获取其大小作为起始位置
    resume_byte_pos = 0
    if os.path.exists(filepath):
        resume_byte_pos = os.path.getsize(filepath)
        headers['Range'] = f'bytes={resume_byte_pos}-'
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # 获取总文件大小
            total_size = int(response.headers.get('content-length', 0))
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])
            
            print(f"下载: {os.path.basename(filepath)} ({total_size} 字节)")
            
            # 以追加模式打开文件
            mode = 'ab' if resume_byte_pos > 0 else 'wb'
            with open(filepath, mode) as f:
                downloaded = resume_byte_pos
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✓ 成功下载: {filepath}")
            return True
            
        except Exception as e:
            retries += 1
            print(f"\n下载失败 (尝试 {retries}/{max_retries}): {e}")
            if retries < max_retries:
                print("等待5秒后重试...")
                time.sleep(5)
    
    print(f"\n✗ 下载失败: {filepath}")
    return False

def download_vaw_annotations():
    """下载VAW数据集标注文件"""
    print("=== 下载VAW数据集标注文件 ===")
    
    # GitHub仓库的数据文件
    vaw_files = {
        'train_part1.json': 'https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part1.json',
        'train_part2.json': 'https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part2.json',
        'val.json': 'https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/val.json',
        'test.json': 'https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/test.json'
    }
    
    vaw_anno_dir = r'D:\KKK\data\VAW\annotations'
    os.makedirs(vaw_anno_dir, exist_ok=True)
    
    success_count = 0
    for filename, url in vaw_files.items():
        filepath = os.path.join(vaw_anno_dir, filename)
        if download_file_with_resume(url, filepath):
            success_count += 1
    
    print(f"VAW标注文件下载完成: {success_count}/{len(vaw_files)} 个文件成功")
    return success_count == len(vaw_files)

def create_sample_files():
    """创建示例文件以验证目录结构"""
    print("=== 创建示例文件验证目录结构 ===")
    
    # 创建VAW示例metadata
    vaw_metadata = {
        "dataset_name": "VAW",
        "version": "1.0",
        "description": "Visual Attributes in the Wild dataset",
        "total_attributes": 620,
        "total_instances": 260895,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(r'D:\KKK\data\VAW\metadata\dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(vaw_metadata, f, indent=2, ensure_ascii=False)
    
    # 创建CelebA示例metadata
    celeba_metadata = {
        "dataset_name": "CelebA",
        "version": "1.0",
        "description": "Large-scale CelebFaces Attributes Dataset",
        "total_images": 202599,
        "total_attributes": 40,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(r'D:\KKK\data\CelebA\metadata\dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(celeba_metadata, f, indent=2, ensure_ascii=False)
    
    print("✓ 示例metadata文件创建完成")

def main():
    """主函数"""
    print("数据集下载脚本启动")
    print("目标目录：D:\\KKK\\data")
    
    # 检查目录是否存在
    if not os.path.exists(r'D:\KKK\data'):
        print("错误：目标目录不存在")
        return
    
    # 下载VAW标注文件
    vaw_success = download_vaw_annotations()
    
    # 创建示例文件
    create_sample_files()
    
    # 总结
    print("\n=== 下载总结 ===")
    print(f"VAW标注文件: {'✓ 成功' if vaw_success else '✗ 失败'}")
    
    print("\n=== 后续步骤 ===")
    print("1. CelebA图像文件需要从Google Drive手动下载：")
    print("   https://drive.google.com/file/d/1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684/view")
    print("2. VAW图像文件来自Visual Genome数据集，根据需要下载")
    print("3. 下载完成后运行解压和组织步骤")

if __name__ == "__main__":
    main() 