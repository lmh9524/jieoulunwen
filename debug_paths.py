#!/usr/bin/env python3
"""CelebA数据集路径诊断脚本"""

import os
import sys

def check_path_exists(path, description):
    """检查路径是否存在并显示详细信息"""
    abs_path = os.path.abspath(path)
    exists = os.path.exists(path)
    
    print(f"🔍 {description}")
    print(f"   相对路径: {path}")
    print(f"   绝对路径: {abs_path}")
    print(f"   是否存在: {'✅' if exists else '❌'}")
    
    if exists and os.path.isdir(path):
        try:
            files = os.listdir(path)
            print(f"   目录内容数量: {len(files)}")
            if len(files) <= 10:
                print(f"   内容: {files}")
            else:
                print(f"   部分内容: {files[:5]} ... (共{len(files)}个)")
        except PermissionError:
            print("   无法读取目录内容（权限不足）")
    
    print()
    return exists

def main():
    print("=" * 60)
    print("CelebA 数据集路径诊断")
    print("=" * 60)
    
    # 显示当前工作目录
    current_dir = os.getcwd()
    print(f"📍 当前工作目录: {current_dir}")
    print()
    
    # 检查各种可能的路径
    possible_paths = [
        ("..", "上级目录"),
        (".", "当前目录"),
        ("/autodl-tmp", "绝对路径 /autodl-tmp"),
        ("~/autodl-tmp", "用户目录下的 autodl-tmp"),
        (os.path.expanduser("~/autodl-tmp"), "展开后的用户目录"),
    ]
    
    valid_roots = []
    
    for path, desc in possible_paths:
        if check_path_exists(path, f"检查根目录: {desc}"):
            valid_roots.append(path)
    
    print("=" * 60)
    print("检查CelebA数据集特定目录")
    print("=" * 60)
    
    for root in valid_roots:
        print(f"🗂️ 在根目录 {root} 中检查CelebA数据:")
        
        img_path = os.path.join(root, "img_align_celeba")
        anno_path = os.path.join(root, "Anno")
        eval_path = os.path.join(root, "Eval")
        
        img_exists = check_path_exists(img_path, "图像目录 (img_align_celeba)")
        anno_exists = check_path_exists(anno_path, "标注目录 (Anno)")
        eval_exists = check_path_exists(eval_path, "评估目录 (Eval)")
        
        if img_exists and anno_exists:
            print(f"🎉 找到完整的CelebA数据集！")
            print(f"   推荐配置: data_path = '{root}'")
            
            # 检查具体的标注文件
            attr_file = os.path.join(anno_path, "list_attr_celeba.txt")
            partition_file = os.path.join(eval_path, "list_eval_partition.txt")  # 修正：在Eval目录中
            
            print(f"   属性文件: {'✅' if os.path.exists(attr_file) else '❌'} {attr_file}")
            print(f"   分割文件: {'✅' if os.path.exists(partition_file) else '❌'} {partition_file}")
            
            # 统计图像数量
            if img_exists:
                try:
                    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"   图像文件数量: {len(img_files)}")
                except:
                    print("   无法统计图像文件数量")
            
            print()
            return root
        
        print("-" * 40)
    
    print("❌ 未找到完整的CelebA数据集")
    return None

if __name__ == "__main__":
    result = main()
    if result:
        sys.exit(0)
    else:
        sys.exit(1) 