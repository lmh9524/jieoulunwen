#!/usr/bin/env python3
"""
数据集完整性检查脚本
检查VAW和CelebA数据集的完整性并生成详细报告
"""

import os
import json
from pathlib import Path

def check_file_exists_and_size(filepath):
    """检查文件是否存在并返回大小信息"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return {
            'exists': True,
            'size': size,
            'size_mb': round(size / (1024*1024), 2),
            'status': '✅' if size > 1000 else '⚠️'
        }
    else:
        return {
            'exists': False,
            'size': 0,
            'size_mb': 0,
            'status': '❌'
        }

def check_vaw_dataset():
    """检查VAW数据集完整性"""
    print("=== VAW数据集完整性检查 ===")
    
    base_dir = Path(r'D:\KKK\data\VAW')
    annotations_dir = base_dir / 'annotations'
    images_dir = base_dir / 'images'
    
    # 检查标注文件
    annotation_files = ['train_part1.json', 'train_part2.json', 'val.json', 'test.json']
    
    for filename in annotation_files:
        filepath = annotations_dir / filename
        info = check_file_exists_and_size(filepath)
        print(f"  {info['status']} {filename}: {info['size_mb']} MB")
        
        # 验证JSON格式
        if info['exists'] and info['size'] > 1000:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"    📊 包含 {len(data)} 条记录")
                    elif isinstance(data, dict):
                        print(f"    📊 包含 {len(data)} 个键")
            except Exception as e:
                print(f"    ⚠️ JSON格式错误: {e}")
    
    # 检查图像目录
    if images_dir.exists():
        image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
        print(f"  📁 图像目录: {image_count} 个图像文件")
    else:
        print("  ❌ 图像目录不存在")

def check_celeba_dataset():
    """检查CelebA数据集完整性"""
    print("\n=== CelebA数据集完整性检查 ===")
    
    base_dir = Path(r'D:\KKK\data\CelebA')
    annotations_dir = base_dir / 'annotations'
    images_dir = base_dir / 'img_align_celeba'
    
    # 检查标注文件
    annotation_files = [
        'list_attr_celeba.txt',
        'list_bbox_celeba.txt', 
        'list_eval_partition.txt'
    ]
    
    for filename in annotation_files:
        filepath = annotations_dir / filename
        info = check_file_exists_and_size(filepath)
        print(f"  {info['status']} {filename}: {info['size_mb']} MB")
        
        # 检查文件内容行数
        if info['exists'] and info['size'] > 1000:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"    📊 包含 {len(lines)} 行")
            except Exception as e:
                print(f"    ⚠️ 读取错误: {e}")
    
    # 检查图像目录
    if images_dir.exists():
        image_files = list(images_dir.glob('*.jpg'))
        print(f"  📁 图像目录: {len(image_files)} 个图像文件")
        
        # 检查预期的图像数量
        if len(image_files) >= 200000:
            print("    ✅ 图像数量正常 (>= 200k)")
        elif len(image_files) >= 100000:
            print("    ⚠️ 图像数量偏少，但可用")
        else:
            print("    ❌ 图像数量严重不足")
    else:
        print("  ❌ 图像目录不存在")
    
    # 检查zip文件
    zip_file = base_dir / 'img_align_celeba.zip'
    if zip_file.exists():
        zip_info = check_file_exists_and_size(zip_file)
        print(f"  📦 原始zip文件: {zip_info['size_mb']} MB")

def generate_summary():
    """生成总结报告"""
    print("\n=== 数据集状态总结 ===")
    
    # VAW检查
    vaw_annotations = Path(r'D:\KKK\data\VAW\annotations')
    vaw_files = ['train_part1.json', 'train_part2.json', 'val.json', 'test.json']
    vaw_complete = all((vaw_annotations / f).exists() and 
                      (vaw_annotations / f).stat().st_size > 1000 
                      for f in vaw_files)
    
    # CelebA检查  
    celeba_annotations = Path(r'D:\KKK\data\CelebA\annotations')
    celeba_files = ['list_attr_celeba.txt', 'list_bbox_celeba.txt', 'list_eval_partition.txt']
    celeba_annotations_complete = all((celeba_annotations / f).exists() and 
                                    (celeba_annotations / f).stat().st_size > 1000 
                                    for f in celeba_files)
    
    celeba_images = Path(r'D:\KKK\data\CelebA\img_align_celeba')
    celeba_images_complete = celeba_images.exists() and len(list(celeba_images.glob('*.jpg'))) > 100000
    
    print(f"VAW数据集标注: {'✅ 完整' if vaw_complete else '⚠️ 不完整'}")
    print(f"CelebA数据集标注: {'✅ 完整' if celeba_annotations_complete else '⚠️ 不完整'}")  
    print(f"CelebA数据集图像: {'✅ 完整' if celeba_images_complete else '⚠️ 不完整'}")
    
    if vaw_complete and celeba_annotations_complete and celeba_images_complete:
        print("\n🎉 所有数据集都已完整准备好！")
    else:
        print("\n⚠️ 部分数据集存在问题，请检查上述详细信息")

if __name__ == "__main__":
    print("开始数据集完整性检查...\n")
    
    check_vaw_dataset()
    check_celeba_dataset() 
    generate_summary()
    
    print("\n检查完成！") 