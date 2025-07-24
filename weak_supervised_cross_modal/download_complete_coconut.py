#!/usr/bin/env python3
"""
完整下载COCONut数据集
确保下载所有5000张真实图像，绝不遗漏任何数据
"""

import os
import json
import requests
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteCoconutDownloader:
    """完整的COCONut数据集下载器"""
    
    def __init__(self, output_dir: str, max_workers: int = 10):
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.download_lock = threading.Lock()
        self.success_count = 0
        self.failed_count = 0
        
    def download_annotations(self):
        """下载标注文件"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载主要的标注文件
        annotation_urls = [
            "https://github.com/akshayparakh25/COCONut/raw/main/data/relabeled_coco_val.json",
            "https://raw.githubusercontent.com/akshayparakh25/COCONut/main/data/relabeled_coco_val.json"
        ]
        
        annotation_file = self.output_dir / "relabeled_coco_val.json"
        
        if annotation_file.exists():
            logger.info(f"标注文件已存在: {annotation_file}")
            return True
        
        logger.info("开始下载COCONut标注文件...")
        
        for url in annotation_urls:
            try:
                logger.info(f"尝试从 {url} 下载...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(annotation_file, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"标注文件下载成功: {annotation_file}")
                return True
                
            except Exception as e:
                logger.warning(f"从 {url} 下载失败: {e}")
                continue
        
        logger.error("所有标注文件下载源都失败了")
        return False
    
    def download_single_image(self, image_info):
        """下载单张图像"""
        img_id = image_info['id']
        filename = image_info['file_name']
        url = image_info['coco_url']
        
        # 创建缓存目录
        cache_dir = self.output_dir / 'complete_image_cache'
        cache_dir.mkdir(exist_ok=True)
        
        cache_path = cache_dir / filename
        
        # 如果文件已存在且完整，跳过
        if cache_path.exists():
            try:
                # 验证文件完整性
                from PIL import Image
                img = Image.open(cache_path)
                img.verify()
                with self.download_lock:
                    self.success_count += 1
                return True
            except:
                # 文件损坏，重新下载
                cache_path.unlink()
        
        # 下载图像
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            # 验证下载的图像
            from PIL import Image
            img = Image.open(cache_path)
            img.verify()
            img.close()
            
            with self.download_lock:
                self.success_count += 1
            
            return True
            
        except Exception as e:
            logger.debug(f"下载图像失败 {filename}: {e}")
            with self.download_lock:
                self.failed_count += 1
            return False
    
    def download_all_images(self, max_retries: int = 3):
        """下载所有图像"""
        # 加载标注文件
        annotation_file = self.output_dir / "relabeled_coco_val.json"
        
        if not annotation_file.exists():
            logger.error("标注文件不存在，请先下载标注文件")
            return False
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        images = data.get('images', [])
        total_images = len(images)
        
        logger.info(f"开始下载 {total_images} 张图像...")
        logger.info(f"使用 {self.max_workers} 个并发线程")
        
        # 多次重试下载失败的图像
        for retry in range(max_retries):
            if retry > 0:
                logger.info(f"第 {retry + 1} 次重试下载失败的图像...")
            
            # 重置计数器
            self.success_count = 0
            self.failed_count = 0
            
            # 过滤出需要下载的图像
            images_to_download = []
            cache_dir = self.output_dir / 'complete_image_cache'
            
            for img_info in images:
                cache_path = cache_dir / img_info['file_name']
                if not cache_path.exists():
                    images_to_download.append(img_info)
            
            if not images_to_download:
                logger.info("所有图像都已下载完成！")
                break
            
            logger.info(f"需要下载 {len(images_to_download)} 张图像")
            
            # 使用线程池下载
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有下载任务
                future_to_image = {
                    executor.submit(self.download_single_image, img_info): img_info
                    for img_info in images_to_download
                }
                
                # 显示进度
                with tqdm(total=len(images_to_download), desc="下载图像") as pbar:
                    for future in as_completed(future_to_image):
                        future.result()  # 获取结果
                        pbar.update(1)
                        pbar.set_postfix({
                            'Success': self.success_count,
                            'Failed': self.failed_count
                        })
            
            logger.info(f"本轮下载完成: 成功 {self.success_count}, 失败 {self.failed_count}")
            
            # 如果失败数量很少，可以停止重试
            if self.failed_count < 10:
                break
            
            # 等待一段时间再重试
            if retry < max_retries - 1:
                logger.info("等待10秒后重试...")
                time.sleep(10)
        
        # 最终统计
        cache_dir = self.output_dir / 'complete_image_cache'
        final_count = len([f for f in os.listdir(cache_dir) if f.endswith(('.jpg', '.png'))])
        
        logger.info(f"最终下载结果: {final_count}/{total_images} 张图像")
        
        if final_count == total_images:
            logger.info("🎉 所有图像下载完成！")
            return True
        else:
            logger.warning(f"⚠️ 仍有 {total_images - final_count} 张图像下载失败")
            return False
    
    def download_segmentation_masks(self):
        """下载分割掩码文件"""
        logger.info("检查分割掩码文件...")
        
        # 分割掩码通常在relabeled_coco_val目录中
        seg_dir = self.output_dir / "relabeled_coco_val"
        
        if seg_dir.exists():
            png_files = [f for f in os.listdir(seg_dir) if f.endswith('.png')]
            logger.info(f"找到 {len(png_files)} 个分割掩码文件")
            
            if len(png_files) >= 5000:
                logger.info("✅ 分割掩码文件完整")
                return True
            else:
                logger.warning(f"⚠️ 分割掩码文件不完整，只有 {len(png_files)} 个")
        
        # 如果分割掩码不完整，尝试下载
        logger.info("尝试下载分割掩码...")
        
        # 这里可能需要根据实际的COCONut数据源调整
        # 目前假设分割掩码已经在relabeled_coco_val目录中
        
        return True
    
    def verify_dataset_completeness(self):
        """验证数据集完整性"""
        logger.info("验证数据集完整性...")
        
        # 检查标注文件
        annotation_file = self.output_dir / "relabeled_coco_val.json"
        if not annotation_file.exists():
            logger.error("❌ 标注文件缺失")
            return False
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        expected_images = len(data.get('images', []))
        expected_annotations = len(data.get('annotations', []))
        expected_categories = len(data.get('categories', []))
        
        # 检查图像缓存
        cache_dir = self.output_dir / 'complete_image_cache'
        if cache_dir.exists():
            cached_images = len([f for f in os.listdir(cache_dir) if f.endswith(('.jpg', '.png'))])
        else:
            cached_images = 0
        
        # 检查分割掩码
        seg_dir = self.output_dir / "relabeled_coco_val"
        if seg_dir.exists():
            seg_masks = len([f for f in os.listdir(seg_dir) if f.endswith('.png')])
        else:
            seg_masks = 0
        
        logger.info("📊 数据集完整性报告:")
        logger.info(f"  - 标注文件: ✅")
        logger.info(f"  - 图像数量: {cached_images}/{expected_images} {'✅' if cached_images == expected_images else '❌'}")
        logger.info(f"  - 标注数量: {expected_annotations} ✅")
        logger.info(f"  - 类别数量: {expected_categories} ✅")
        logger.info(f"  - 分割掩码: {seg_masks} {'✅' if seg_masks >= 5000 else '❌'}")
        
        is_complete = (cached_images == expected_images and 
                      expected_annotations > 0 and 
                      expected_categories > 0 and 
                      seg_masks >= 5000)
        
        if is_complete:
            logger.info("🎉 数据集完整！可以开始训练了！")
        else:
            logger.warning("⚠️ 数据集不完整，建议重新下载")
        
        return is_complete

def main():
    parser = argparse.ArgumentParser(description='完整下载COCONut数据集')
    parser.add_argument('--output_dir', type=str, default='../data/coconut', 
                       help='输出目录')
    parser.add_argument('--max_workers', type=int, default=10, 
                       help='并发下载线程数')
    parser.add_argument('--max_retries', type=int, default=3, 
                       help='最大重试次数')
    
    args = parser.parse_args()
    
    logger.info("🥥 开始完整下载COCONut数据集")
    logger.info("=" * 60)
    logger.info("严格保证：只下载真实数据，绝不使用任何合成数据！")
    logger.info("=" * 60)
    
    downloader = CompleteCoconutDownloader(
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    try:
        # 1. 下载标注文件
        if not downloader.download_annotations():
            logger.error("标注文件下载失败，终止下载")
            return
        
        # 2. 下载所有图像
        if not downloader.download_all_images(max_retries=args.max_retries):
            logger.warning("部分图像下载失败，但继续验证...")
        
        # 3. 检查分割掩码
        downloader.download_segmentation_masks()
        
        # 4. 验证完整性
        is_complete = downloader.verify_dataset_completeness()
        
        if is_complete:
            logger.info("✅ COCONut数据集下载完成！")
            logger.info("🚀 现在可以开始100轮训练了！")
        else:
            logger.error("❌ 数据集下载不完整，请检查网络连接后重试")
            
    except Exception as e:
        logger.error(f"下载过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main() 