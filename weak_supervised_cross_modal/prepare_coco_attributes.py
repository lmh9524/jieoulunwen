"""
Prepare COCO Attributes dataset for training.

This script prepares the COCO Attributes dataset for training by:
1. Generating mock COCO Attributes data
2. Generating mock images
3. Setting up directory structure
"""
import os
import argparse
import subprocess
import sys

def run_command(command):
    """Run a command and print its output."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    return True

def prepare_dataset(data_dir, num_samples=1000, num_images=100):
    """
    Prepare the COCO Attributes dataset.
    
    Args:
        data_dir: Root directory for the dataset
        num_samples: Number of attribute samples to generate
        num_images: Number of images to generate
    """
    print(f"Preparing COCO Attributes dataset in {data_dir}")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate mock COCO Attributes data
    attributes_file = os.path.join(data_dir, 'mock_coco_attributes.pkl')
    cmd_generate_attributes = f"{sys.executable} {os.path.join('data', 'generate_mock_coco_attributes.py')} --output {attributes_file} --num_samples {num_samples}"
    if not run_command(cmd_generate_attributes):
        return False
    
    # Generate mock images
    coconut_dir = os.path.join(data_dir, 'coconut')
    cmd_generate_images = f"{sys.executable} {os.path.join('data', 'generate_mock_images.py')} --output_dir {coconut_dir} --num_images {num_images}"
    if not run_command(cmd_generate_images):
        return False
    
    print("\n" + "="*80)
    print(f"COCO Attributes dataset preparation completed successfully!")
    print(f"Dataset location: {data_dir}")
    print(f"Attributes file: {attributes_file}")
    print(f"Image directory: {coconut_dir}")
    print("="*80)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare COCO Attributes dataset')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root directory for the dataset')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of attribute samples to generate')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to generate')
    
    args = parser.parse_args()
    prepare_dataset(args.data_dir, args.num_samples, args.num_images) 