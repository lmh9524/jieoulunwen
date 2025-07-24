"""
Generate mock COCO Attributes dataset for testing.

This script creates a mock version of the COCO Attributes dataset in the format
expected by the COCOAttributesDataset class in dataset_adapters.py.
"""
import os
import pickle
import numpy as np
import random
from tqdm import tqdm
import argparse

def generate_mock_coco_attributes(output_file, num_samples=1000):
    """
    Generate a mock COCO Attributes dataset and save it to a pickle file.
    
    Args:
        output_file: Path to save the generated dataset.
        num_samples: Number of samples to generate.
    """
    print(f"Generating mock COCO Attributes dataset with {num_samples} samples...")
    
    # Create mock data structure
    mock_data = {
        'ann_vecs': {},           # patch_id -> attribute vector
        'patch_id_to_ann_id': {}, # patch_id -> annotation_id
        'split': {},              # patch_id -> split name
        'attributes': []          # list of attribute definitions
    }
    
    # Generate 204 mock attributes
    for i in range(204):
        attr_type = "color" if i < 20 else \
                   "material" if i < 60 else \
                   "shape" if i < 100 else \
                   "texture" if i < 140 else \
                   "size" if i < 180 else "other"
        
        mock_data['attributes'].append({
            'id': i,
            'name': f'attr_{i}',
            'type': attr_type
        })
    
    # Generate mock samples
    for i in tqdm(range(num_samples)):
        patch_id = i
        ann_id = 10000 + i
        
        # Generate a random binary attribute vector (204 dimensions)
        attr_vec = np.random.randint(0, 2, size=204).astype(np.float32)
        
        # Assign to train or validation split
        split = 'train2014' if random.random() < 0.8 else 'val2014'
        
        mock_data['ann_vecs'][patch_id] = attr_vec
        mock_data['patch_id_to_ann_id'][patch_id] = ann_id
        mock_data['split'][patch_id] = split
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save the mock dataset
    with open(output_file, 'wb') as f:
        pickle.dump(mock_data, f)
    
    print(f"Mock COCO Attributes dataset saved to {output_file}")
    print(f"Dataset contains {len(mock_data['ann_vecs'])} samples and {len(mock_data['attributes'])} attributes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate mock COCO Attributes dataset')
    parser.add_argument('--output', type=str, default='./data/mock_coco_attributes.pkl',
                        help='Path to save the generated dataset')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    
    args = parser.parse_args()
    generate_mock_coco_attributes(args.output, args.num_samples) 