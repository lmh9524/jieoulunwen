import os
from pathlib import Path

# 检查CelebA标注文件
celeba_annotations = Path(r'D:\KKK\data\CelebA\annotations')
print("=== CelebA标注文件检查 ===")

files_to_check = [
    'list_attr_celeba.txt',
    'list_bbox_celeba.txt', 
    'list_eval_partition.txt'
]

for filename in files_to_check:
    filepath = celeba_annotations / filename
    if filepath.exists():
        size = filepath.stat().st_size
        size_mb = round(size / (1024*1024), 2)
        status = '✅' if size > 1000 else '⚠️'
        print(f"{status} {filename}: {size_mb} MB")
    else:
        print(f"❌ {filename}: 文件不存在")

# 检查CelebA图像目录
celeba_images = Path(r'D:\KKK\data\CelebA\img_align_celeba')
print(f"\n=== CelebA图像目录检查 ===")
if celeba_images.exists():
    image_files = list(celeba_images.glob('*.jpg'))
    print(f"✅ 图像目录存在，包含 {len(image_files)} 个图像文件")
else:
    print("❌ 图像目录不存在")

# 检查VAW标注文件
vaw_annotations = Path(r'D:\KKK\data\VAW\annotations')
print(f"\n=== VAW标注文件检查 ===")

vaw_files = ['train_part1.json', 'train_part2.json', 'val.json', 'test.json']
for filename in vaw_files:
    filepath = vaw_annotations / filename
    if filepath.exists():
        size = filepath.stat().st_size
        size_mb = round(size / (1024*1024), 2)
        status = '✅' if size > 1000 else '⚠️'
        print(f"{status} {filename}: {size_mb} MB")
    else:
        print(f"❌ {filename}: 文件不存在")

print("\n检查完成！") 