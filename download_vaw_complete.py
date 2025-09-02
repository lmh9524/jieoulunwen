#!/usr/bin/env python3
"""
完整VAW数据集下载脚本
包括VAW标注文件和Visual Genome图像的下载
"""

import os
import requests
import json
import zipfile
from pathlib import Path
import time

def download_file_with_progress(url, filepath, chunk_size=8192):
    """带进度显示的文件下载"""
    print(f"下载: {url}")
    print(f"保存到: {filepath}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r进度: {percent:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)", end='')
        
        print(f"\n下载完成: {filepath}")
        return True
        
    except Exception as e:
        print(f"\n下载失败: {e}")
        return False

def download_vaw_annotations():
    """下载VAW标注文件"""
    print("=== 下载VAW标注文件 ===")
    
    base_url = "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data"
    annotations_dir = Path("D:/KKK/data/VAW/annotations")
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = [
        "train_part1.json",
        "train_part2.json", 
        "val.json",
        "test.json"
    ]
    
    success_count = 0
    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        filepath = annotations_dir / filename
        
        if download_file_with_progress(url, filepath):
            success_count += 1
        time.sleep(1)  # 避免请求过快
    
    print(f"\nVAW标注文件下载完成: {success_count}/{len(files_to_download)}")
    return success_count == len(files_to_download)

def get_vaw_image_ids():
    """从VAW标注文件中提取所需的图像ID"""
    print("=== 提取VAW所需的图像ID ===")
    
    annotations_dir = Path("D:/KKK/data/VAW/annotations")
    image_ids = set()
    
    for json_file in ["train_part1.json", "train_part2.json", "val.json", "test.json"]:
        filepath = annotations_dir / json_file
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if 'image_id' in item:
                            image_ids.add(item['image_id'])
                print(f"从 {json_file} 提取了 {len([item for item in data if 'image_id' in item])} 个图像ID")
            except Exception as e:
                print(f"读取 {json_file} 失败: {e}")
    
    print(f"总共需要下载 {len(image_ids)} 个唯一图像")
    return list(image_ids)

def download_visual_genome_images(image_ids, max_images=1000):
    """下载Visual Genome图像（限制数量避免过大）"""
    print(f"=== 下载Visual Genome图像 (前{max_images}个) ===")
    
    images_dir = Path("D:/KKK/data/VAW/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 限制下载数量
    image_ids = image_ids[:max_images]
    
    base_url_1 = "https://cs.stanford.edu/people/rak248/VG_100K"
    base_url_2 = "https://cs.stanford.edu/people/rak248/VG_100K_2"
    
    success_count = 0
    for i, image_id in enumerate(image_ids):
        print(f"\n下载图像 {i+1}/{len(image_ids)}: {image_id}")
        
        filename = f"{image_id}.jpg"
        filepath = images_dir / filename
        
        if filepath.exists():
            print(f"图像已存在，跳过: {filename}")
            success_count += 1
            continue
        
        # 尝试两个URL
        urls = [f"{base_url_1}/{filename}", f"{base_url_2}/{filename}"]
        
        downloaded = False
        for url in urls:
            if download_file_with_progress(url, filepath):
                success_count += 1
                downloaded = True
                break
        
        if not downloaded:
            print(f"无法下载图像: {image_id}")
        
        time.sleep(0.5)  # 避免请求过快
    
    print(f"\nVisual Genome图像下载完成: {success_count}/{len(image_ids)}")
    return success_count

def create_dataset_summary():
    """创建数据集摘要"""
    print("=== 创建数据集摘要 ===")
    
    vaw_dir = Path("D:/KKK/data/VAW")
    annotations_dir = vaw_dir / "annotations"
    images_dir = vaw_dir / "images"
    
    summary = {
        "dataset": "VAW (Visual Attributes in the Wild)",
        "annotations": {},
        "images": {
            "total_downloaded": 0,
            "files": []
        }
    }
    
    # 检查标注文件
    for json_file in ["train_part1.json", "train_part2.json", "val.json", "test.json"]:
        filepath = annotations_dir / json_file
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary["annotations"][json_file] = {
                        "size_mb": round(size_mb, 2),
                        "records": len(data),
                        "status": "完整"
                    }
            except:
                summary["annotations"][json_file] = {
                    "size_mb": round(size_mb, 2),
                    "records": 0,
                    "status": "损坏"
                }
    
    # 检查图像文件
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg"))
        summary["images"]["total_downloaded"] = len(image_files)
        summary["images"]["files"] = [f.name for f in image_files[:10]]  # 只显示前10个
    
    # 保存摘要
    summary_file = vaw_dir / "dataset_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"数据集摘要已保存到: {summary_file}")
    return summary

def main():
    """主函数"""
    print("开始VAW数据集完整下载...\n")
    
    # 1. 下载VAW标注文件
    if not download_vaw_annotations():
        print("VAW标注文件下载失败，终止程序")
        return
    
    # 2. 提取所需图像ID
    image_ids = get_vaw_image_ids()
    if not image_ids:
        print("无法提取图像ID，终止程序")
        return
    
    # 3. 下载部分Visual Genome图像（避免下载过多）
    max_images = min(1000, len(image_ids))  # 最多下载1000张图像
    print(f"\n将下载前 {max_images} 张图像作为示例...")
    download_visual_genome_images(image_ids, max_images)
    
    # 4. 创建数据集摘要
    summary = create_dataset_summary()
    
    print("\n=== VAW数据集下载完成 ===")
    print(f"标注文件: {len(summary['annotations'])} 个")
    print(f"图像文件: {summary['images']['total_downloaded']} 个")
    print("\n如需下载更多图像，请修改 max_images 参数")

if __name__ == "__main__":
    main() 