#!/usr/bin/env python3
"""
数据集验证脚本
检查下载的VAW和CelebA数据集文件完整性并生成报告
"""

import os
import json
import time
from pathlib import Path

def check_vaw_files():
    """检查VAW数据集文件"""
    print("=== 检查VAW数据集 ===")
    
    vaw_dir = Path(r'D:\KKK\data\VAW')
    annotations_dir = vaw_dir / 'annotations'
    
    expected_files = ['train_part1.json', 'train_part2.json', 'val.json', 'test.json']
    results = {}
    
    for filename in expected_files:
        filepath = annotations_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            results[filename] = {
                'exists': True,
                'size': size,
                'size_mb': round(size / (1024*1024), 2)
            }
            
            # 验证JSON格式
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results[filename]['valid_json'] = True
                results[filename]['records'] = len(data) if isinstance(data, list) else 'unknown'
            except Exception as e:
                results[filename]['valid_json'] = False
                results[filename]['error'] = str(e)
        else:
            results[filename] = {'exists': False}
    
    return results

def check_celeba_files():
    """检查CelebA数据集文件"""
    print("=== 检查CelebA数据集 ===")
    
    celeba_dir = Path(r'D:\KKK\data\CelebA')
    
    results = {}
    
    # 检查图像zip文件
    img_zip = celeba_dir / 'img_align_celeba.zip'
    if img_zip.exists():
        size = img_zip.stat().st_size
        results['img_align_celeba.zip'] = {
            'exists': True,
            'size': size,
            'size_mb': round(size / (1024*1024), 2),
            'expected_size_gb': 1.3,
            'download_complete': size > 1000000000  # 大于1GB认为下载较完整
        }
    else:
        results['img_align_celeba.zip'] = {'exists': False}
    
    return results

def generate_dataset_report():
    """生成数据集状态报告"""
    print("数据集验证报告")
    print("=" * 50)
    
    # 检查VAW
    vaw_results = check_vaw_files()
    print("\nVAW数据集状态:")
    total_vaw = len(vaw_results)
    success_vaw = sum(1 for r in vaw_results.values() if r.get('exists', False))
    
    for filename, result in vaw_results.items():
        if result.get('exists', False):
            status = "✓" if result.get('valid_json', False) else "⚠"
            size_info = f"({result.get('size_mb', 0)} MB)"
            records = result.get('records', 'N/A')
            print(f"  {status} {filename} {size_info} - {records} 条记录")
        else:
            print(f"  ✗ {filename} - 文件不存在")
    
    print(f"\nVAW总结: {success_vaw}/{total_vaw} 个文件可用")
    
    # 检查CelebA
    celeba_results = check_celeba_files()
    print("\nCelebA数据集状态:")
    
    for filename, result in celeba_results.items():
        if result.get('exists', False):
            size_mb = result.get('size_mb', 0)
            complete = result.get('download_complete', False)
            status = "✓" if complete else "⚠"
            print(f"  {status} {filename} ({size_mb} MB) - {'完整' if complete else '下载未完成'}")
        else:
            print(f"  ✗ {filename} - 文件不存在")
    
    # 目录结构检查
    print("\n目录结构检查:")
    dirs_to_check = [
        r'D:\KKK\data\VAW\annotations',
        r'D:\KKK\data\VAW\images', 
        r'D:\KKK\data\VAW\metadata',
        r'D:\KKK\data\CelebA\img_align_celeba',
        r'D:\KKK\data\CelebA\annotations',
        r'D:\KKK\data\CelebA\metadata'
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path}")
    
    # 推荐后续步骤
    print("\n=== 推荐后续步骤 ===")
    
    if success_vaw < total_vaw:
        print("1. 继续下载缺失的VAW标注文件")
    
    celeba_zip_complete = celeba_results.get('img_align_celeba.zip', {}).get('download_complete', False)
    if not celeba_zip_complete:
        print("2. 重新下载CelebA图像文件 (img_align_celeba.zip)")
        print("   建议使用: python -m gdown 1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684 -O D:\\KKK\\data\\CelebA\\img_align_celeba.zip")
    
    if celeba_zip_complete:
        print("3. 解压CelebA图像文件")
        print("4. 下载CelebA标注文件")
    
    print("5. 下载VAW相关的Visual Genome图像 (可选，按需)")

def main():
    """主函数"""
    print("数据集验证工具启动")
    print(f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    generate_dataset_report()

if __name__ == "__main__":
    main() 