"""
Generate mock image files for COCO Attributes dataset.

This script creates mock image files for testing the COCOAttributesDataset class.
"""
import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def generate_mock_images(output_dir, num_images=100):
    """
    Generate mock image files for testing.
    
    Args:
        output_dir: Directory to save the generated images.
        num_images: Number of images to generate.
    """
    print(f"Generating {num_images} mock images...")
    
    # Create directories for complete_image_cache and relabeled_coco_val
    complete_image_cache_dir = os.path.join(output_dir, 'complete_image_cache')
    relabeled_coco_val_dir = os.path.join(output_dir, 'relabeled_coco_val')
    
    os.makedirs(complete_image_cache_dir, exist_ok=True)
    os.makedirs(relabeled_coco_val_dir, exist_ok=True)
    
    # Generate mock images
    for i in tqdm(range(num_images)):
        # Create a random RGB image (224x224)
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save with different formats for testing path resolution in the dataset
        ann_id = 10000 + i
        
        # Format 1: 12-digit zero-padded
        img.save(os.path.join(complete_image_cache_dir, f'{ann_id:012d}.jpg'))
        
        # Format 2: 6-digit zero-padded (for some images)
        if i % 3 == 0:
            img.save(os.path.join(complete_image_cache_dir, f'{ann_id:06d}.jpg'))
        
        # Format 3: PNG in relabeled_coco_val (for some images)
        if i % 5 == 0:
            img.save(os.path.join(relabeled_coco_val_dir, f'{ann_id:012d}.png'))
    
    print(f"Mock images saved to {output_dir}")
    print(f"- {complete_image_cache_dir}: {len(os.listdir(complete_image_cache_dir))} images")
    print(f"- {relabeled_coco_val_dir}: {len(os.listdir(relabeled_coco_val_dir))} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate mock images for COCO Attributes dataset')
    parser.add_argument('--output_dir', type=str, default='./data/coconut',
                        help='Directory to save the generated images')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to generate')
    
    args = parser.parse_args()
    generate_mock_images(args.output_dir, args.num_images) 