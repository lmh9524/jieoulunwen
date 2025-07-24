
import os
import json
import joblib
import sys

# 将 cocottributes 目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cocottributes-master', 'cocottributes-master')))

from utils.utils import convert

def main():
    """
    生成 cocottributes_new_version.jbl 文件。
    """
    # 定义文件路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 假设 COCO 标注文件位于 data/COCO_Dataset/annotations/
    train_annotations_file = os.path.join(base_dir, 'data', 'COCO_Dataset', 'annotations', 'instances_train2017.json')
    val_annotations_file = os.path.join(base_dir, 'data', 'COCO_Dataset', 'annotations', 'instances_val2017.json')
    
    # COCO Attributes 的原始数据文件
    original_attributes_file = os.path.join(base_dir, 'weak_supervised_cross_modal', 'data', 'mock_coco_attributes.pkl')
    
    # 目标输出文件
    output_jbl_file = os.path.join(base_dir, 'data', 'cocottributes-master', 'cocottributes-master', 'MSCOCO', 'cocottributes_new_version.jbl')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jbl_file), exist_ok=True)
    
    print("开始生成 COCO Attributes jbl 文件...")
    
    # 检查输入文件是否存在
    if not os.path.exists(train_annotations_file):
        print(f"错误: 训练集标注文件未找到: {train_annotations_file}")
        print("请将 instances_train2017.json 放置在 data/COCO_Dataset/annotations/ 目录下。")
        return

    if not os.path.exists(val_annotations_file):
        print(f"错误: 验证集标注文件未找到: {val_annotations_file}")
        print("请将 instances_val2017.json 放置在 data/COCO_Dataset/annotations/ 目录下。")
        return

    if not os.path.exists(original_attributes_file):
        print(f"错误: COCO Attributes 原始数据文件未找到: {original_attributes_file}")
        print("请将原始的属性数据文件 (.pkl) 放置在 data/cocottributes-master/cocottributes-master/ 目录下，并更新脚本中的文件名。")
        return
        
    # 加载数据
    print("加载原始数据...")
    with open(train_annotations_file, 'r') as f:
        train_annotations = json.load(f)['annotations']
        
    with open(val_annotations_file, 'r') as f:
        val_annotations = json.load(f)['annotations']

    with open(original_attributes_file, 'rb') as f:
        attributes = joblib.load(f) # 假设原始文件也是 joblib 格式
    
    # 调用 convert 函数
    print("转换数据格式...")
    # 修改convert函数的工作目录，确保相对路径正确
    current_dir = os.getcwd()
    utils_dir = os.path.join(base_dir, 'data', 'cocottributes-master', 'cocottributes-master', 'utils')
    os.chdir(utils_dir)
    convert(train_annotations, val_annotations, attributes)
    os.chdir(current_dir)
    
    print(f"成功生成 jbl 文件: {output_jbl_file}")
    
if __name__ == "__main__":
    main() 