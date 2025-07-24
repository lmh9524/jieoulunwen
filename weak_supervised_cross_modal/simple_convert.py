import os
import joblib
import numpy as np
import pickle

def main():
    """
    直接将mock_coco_attributes.pkl转换为cocottributes_new_version.jbl格式
    """
    # 定义文件路径
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # 输入文件
    mock_file = os.path.join(base_dir, 'data', 'mock_coco_attributes.pkl')
    
    # 输出文件
    output_dir = os.path.join(project_root, 'data', 'cocottributes-master', 'cocottributes-master', 'MSCOCO')
    output_file = os.path.join(output_dir, 'cocottributes_new_version.jbl')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始转换: {mock_file} -> {output_file}")
    
    # 加载mock数据
    try:
        with open(mock_file, 'rb') as f:
            mock_data = pickle.load(f)
        
        print(f"成功加载mock数据: {len(mock_data['ann_vecs'])} 个样本, {len(mock_data['attributes'])} 个属性")
        
        # 创建新的数据结构 (Varun's Implementation格式)
        new_data = {
            'attributes': sorted(mock_data['attributes'], key=lambda x: x['id']),
            'ann_attrs': {}
        }
        
        # 转换数据
        for patch_id, attrs_vector in mock_data['ann_vecs'].items():
            ann_id = mock_data['patch_id_to_ann_id'][patch_id]
            split = mock_data['split'][patch_id]
            
            new_data['ann_attrs'][str(ann_id)] = {
                'attrs_vector': attrs_vector,
                'split': split
            }
        
        # 保存为joblib文件
        joblib.dump(new_data, output_file, compress=3)
        print(f"成功生成文件: {output_file}")
        
        # 为了方便调试，也保存一份到weak_supervised_cross_modal目录
        debug_output = os.path.join(base_dir, 'cocottributes_new_version.jbl')
        joblib.dump(new_data, debug_output, compress=3)
        print(f"额外保存了一份到: {debug_output}")
        
        return True
    
    except Exception as e:
        print(f"转换失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n转换成功完成!")
    else:
        print("\n转换失败!") 