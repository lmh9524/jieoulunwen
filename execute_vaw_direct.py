#!/usr/bin/env python3
"""
[执行模式] VAW数据集直接下载执行脚本
严格按照计划执行VAW数据集重新下载
"""

import os
import urllib.request
import json
from pathlib import Path
import time

def create_directories():
    """创建目录结构"""
    print("[执行模式] 步骤1: 创建目录结构")
    
    directories = [
        "D:/KKK/data/VAW/annotations",
        "D:/KKK/data/VAW/images", 
        "D:/KKK/data/VAW/metadata"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    print("目录结构创建完成\n")

def download_file(url, filepath, description=""):
    """下载文件"""
    print(f"下载: {description}")
    print(f"URL: {url}")
    print(f"保存到: {filepath}")
    
    try:
        # 添加User-Agent避免被拒绝
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        
        # 检查文件大小
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print(f"✅ 下载完成: {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_vaw_annotations():
    """下载VAW标注文件"""
    print("[执行模式] 步骤2: 下载VAW标注文件")
    
    base_url = "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data"
    files_to_download = [
        ("train_part1.json", "训练集第1部分"),
        ("train_part2.json", "训练集第2部分"),
        ("val.json", "验证集"),
        ("test.json", "测试集")
    ]
    
    success_count = 0
    for filename, description in files_to_download:
        url = f"{base_url}/{filename}"
        filepath = f"D:/KKK/data/VAW/annotations/{filename}"
        
        if download_file(url, filepath, f"{description} ({filename})"):
            success_count += 1
        
        print()  # 空行
        time.sleep(1)  # 避免请求过快
    
    print(f"VAW标注文件下载完成: {success_count}/{len(files_to_download)}\n")
    return success_count == len(files_to_download)

def download_sample_images():
    """下载样本Visual Genome图像"""
    print("[执行模式] 步骤3: 下载样本Visual Genome图像")
    
    sample_images = [
        ("1.jpg", "https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg"),
        ("2.jpg", "https://cs.stanford.edu/people/rak248/VG_100K_2/2.jpg"),
        ("3.jpg", "https://cs.stanford.edu/people/rak248/VG_100K/3.jpg"),
        ("4.jpg", "https://cs.stanford.edu/people/rak248/VG_100K/4.jpg"),
        ("5.jpg", "https://cs.stanford.edu/people/rak248/VG_100K_2/5.jpg")
    ]
    
    success_count = 0
    for filename, url in sample_images:
        filepath = f"D:/KKK/data/VAW/images/{filename}"
        
        if download_file(url, filepath, f"样本图像 {filename}"):
            success_count += 1
        
        print()  # 空行
        time.sleep(0.5)  # 避免请求过快
    
    print(f"样本图像下载完成: {success_count}/{len(sample_images)}\n")
    return success_count

def verify_downloads():
    """验证下载结果"""
    print("[执行模式] 步骤4: 验证下载结果")
    
    # 检查标注文件
    annotations_dir = Path("D:/KKK/data/VAW/annotations")
    print("标注文件检查:")
    for json_file in ["train_part1.json", "train_part2.json", "val.json", "test.json"]:
        filepath = annotations_dir / json_file
        if filepath.exists():
            size = filepath.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  ✅ {json_file}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {json_file}: 文件不存在")
    
    # 检查图像文件
    images_dir = Path("D:/KKK/data/VAW/images")
    print("\n图像文件检查:")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg"))
        print(f"  ✅ 图像目录存在，包含 {len(image_files)} 个文件")
        for img in image_files:
            size_kb = img.stat().st_size / 1024
            print(f"    - {img.name}: {size_kb:.1f} KB")
    else:
        print("  ❌ 图像目录不存在")

def create_execution_report():
    """创建执行报告"""
    print("\n[执行模式] 步骤5: 创建执行报告")
    
    report = {
        "execution_mode": "VAW数据集重新下载",
        "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "执行完成",
        "annotations": {},
        "images": {}
    }
    
    # 统计标注文件
    annotations_dir = Path("D:/KKK/data/VAW/annotations")
    for json_file in ["train_part1.json", "train_part2.json", "val.json", "test.json"]:
        filepath = annotations_dir / json_file
        if filepath.exists():
            size = filepath.stat().st_size
            report["annotations"][json_file] = {
                "size_mb": round(size / (1024 * 1024), 2),
                "status": "下载完成"
            }
        else:
            report["annotations"][json_file] = {
                "size_mb": 0,
                "status": "下载失败"
            }
    
    # 统计图像文件
    images_dir = Path("D:/KKK/data/VAW/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg"))
        report["images"] = {
            "total_count": len(image_files),
            "files": [f.name for f in image_files]
        }
    
    # 保存报告
    report_file = Path("D:/KKK/data/VAW/execution_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"执行报告已保存: {report_file}")
    return report

def main():
    """主执行函数"""
    print("="*50)
    print("[执行模式] VAW数据集重新下载")
    print("="*50)
    
    try:
        # 步骤1: 创建目录
        create_directories()
        
        # 步骤2: 下载标注文件
        if not download_vaw_annotations():
            print("⚠️  部分标注文件下载失败，但继续执行")
        
        # 步骤3: 下载样本图像
        image_count = download_sample_images()
        print(f"已下载 {image_count} 个样本图像")
        
        # 步骤4: 验证下载
        verify_downloads()
        
        # 步骤5: 创建报告
        report = create_execution_report()
        
        print("\n" + "="*50)
        print("[执行模式] VAW数据集重新下载完成")
        print("="*50)
        print("✅ 目录结构已创建")
        print(f"✅ 标注文件: {len([k for k, v in report['annotations'].items() if v['status'] == '下载完成'])}/4 个完成")
        print(f"✅ 样本图像: {report['images']['total_count']} 个完成")
        print("\n请运行验证脚本检查数据集完整性")
        
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误: {e}")
        print("请检查网络连接和目标目录权限")

if __name__ == "__main__":
    main() 