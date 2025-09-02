#!/usr/bin/env python3
"""简单的数据集检查"""

import os

def check_file(path):
    if os.path.exists(path):
        size = os.path.getsize(path)
        return f"✅ {size/1024/1024:.1f} MB"
    else:
        return "❌ 不存在"

print("=== 数据集完整性检查 ===")

# CelebA 标注文件
print("\nCelebA 标注文件:")
print(f"list_attr_celeba.txt: {check_file(r'D:\KKK\data\CelebA\annotations\list_attr_celeba.txt')}")
print(f"list_bbox_celeba.txt: {check_file(r'D:\KKK\data\CelebA\annotations\list_bbox_celeba.txt')}")
print(f"list_eval_partition.txt: {check_file(r'D:\KKK\data\CelebA\annotations\list_eval_partition.txt')}")

# CelebA 图像目录
img_dir = r'D:\KKK\data\CelebA\img_align_celeba'
if os.path.exists(img_dir):
    img_count = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    print(f"CelebA 图像目录: ✅ {img_count} 个图像文件")
else:
    print("CelebA 图像目录: ❌ 不存在")

# VAW 标注文件
print("\nVAW 标注文件:")
print(f"train_part1.json: {check_file(r'D:\KKK\data\VAW\annotations\train_part1.json')}")
print(f"train_part2.json: {check_file(r'D:\KKK\data\VAW\annotations\train_part2.json')}")
print(f"val.json: {check_file(r'D:\KKK\data\VAW\annotations\val.json')}")
print(f"test.json: {check_file(r'D:\KKK\data\VAW\annotations\test.json')}")

print("\n检查完成！") 