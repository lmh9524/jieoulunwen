#!/usr/bin/env python3
"""
Visual Genome数据集完整下载脚本
支持VG_100K和VG_100K_2的批量下载
"""

import os
import sys
import time
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import json
from tqdm import tqdm

class VisualGenomeDownloader:
    def __init__(self, base_dir="D:/KKK/data/VAW/images"):
        self.base_dir = Path(base_dir)
        self.vg_100k_dir = self.base_dir / "VG_100K"
        self.vg_100k_2_dir = self.base_dir / "VG_100K_2"
        
        # 下载源配置
        self.base_urls = [
            "https://cs.stanford.edu/people/rak248/VG_100K/",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/"
        ]
        
        # 下载配置
        self.max_workers = 8
        self.timeout = 30
        self.max_retries = 3
        self.chunk_size = 8192
        
        # 统计信息
        self.stats = {
            "total_files": 0,
            "downloaded": 0,
            "failed": 0,
            "skipped": 0,
            "total_size": 0
        }
        
        # 创建目录
        self.vg_100k_dir.mkdir(parents=True, exist_ok=True)
        self.vg_100k_2_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Visual Genome下载器初始化完成")
        print(f"VG_100K目录: {self.vg_100k_dir}")
        print(f"VG_100K_2目录: {self.vg_100k_2_dir}")
    
    def get_image_list(self, dataset_part):
        """获取需要下载的图像列表"""
        if dataset_part == "VG_100K":
            # VG_100K: 图像ID从1到约50000
            return list(range(1, 50001))
        else:
            # VG_100K_2: 图像ID从2000001到约2058000
            return list(range(2000001, 2058001))
    
    def download_single_image(self, image_id, dataset_part, base_url):
        """下载单个图像文件"""
        filename = f"{image_id}.jpg"
        
        if dataset_part == "VG_100K":
            local_path = self.vg_100k_dir / filename
        else:
            local_path = self.vg_100k_2_dir / filename
        
        # 检查文件是否已存在
        if local_path.exists() and local_path.stat().st_size > 0:
            self.stats["skipped"] += 1
            return {"status": "skipped", "file": filename, "size": local_path.stat().st_size}
        
        # 构建下载URL
        image_url = urljoin(base_url, filename)
        
        # 尝试下载
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    image_url, 
                    timeout=self.timeout,
                    stream=True,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                
                if response.status_code == 200:
                    # 下载文件
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(local_path, 'wb') as f:
                        downloaded_size = 0
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                    
                    # 验证下载完整性
                    if total_size > 0 and downloaded_size != total_size:
                        raise Exception(f"下载不完整: {downloaded_size}/{total_size}")
                    
                    self.stats["downloaded"] += 1
                    self.stats["total_size"] += downloaded_size
                    
                    return {
                        "status": "success", 
                        "file": filename, 
                        "size": downloaded_size,
                        "attempts": attempt + 1
                    }
                
                elif response.status_code == 404:
                    # 文件不存在，跳过
                    self.stats["skipped"] += 1
                    return {"status": "not_found", "file": filename}
                
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.stats["failed"] += 1
                    return {"status": "failed", "file": filename, "error": str(e)}
                else:
                    time.sleep(2 ** attempt)  # 指数退避
        
        return {"status": "failed", "file": filename, "error": "Max retries exceeded"}
    
    def download_dataset_part(self, dataset_part):
        """下载数据集的一部分（VG_100K或VG_100K_2）"""
        print(f"\n开始下载 {dataset_part}...")
        
        image_list = self.get_image_list(dataset_part)
        base_url = self.base_urls[0 if dataset_part == "VG_100K" else 1]
        
        self.stats["total_files"] += len(image_list)
        
        # 创建进度条
        pbar = tqdm(
            total=len(image_list),
            desc=f"下载 {dataset_part}",
            unit="files",
            ncols=100
        )
        
        # 多线程下载
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有下载任务
            future_to_id = {
                executor.submit(self.download_single_image, img_id, dataset_part, base_url): img_id
                for img_id in image_list
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_id):
                result = future.result()
                pbar.update(1)
                
                # 更新进度条描述
                if result["status"] == "success":
                    pbar.set_postfix({
                        "下载": self.stats["downloaded"],
                        "跳过": self.stats["skipped"],
                        "失败": self.stats["failed"]
                    })
        
        pbar.close()
        print(f"{dataset_part} 下载完成!")
    
    def download_all(self):
        """下载完整的Visual Genome数据集"""
        print("=" * 60)
        print("Visual Genome数据集下载开始")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 下载VG_100K
            self.download_dataset_part("VG_100K")
            
            # 下载VG_100K_2
            self.download_dataset_part("VG_100K_2")
            
        except KeyboardInterrupt:
            print("\n下载被用户中断")
            return False
        except Exception as e:
            print(f"\n下载过程中发生错误: {e}")
            return False
        
        # 计算总用时
        total_time = time.time() - start_time
        
        # 生成下载报告
        self.generate_report(total_time)
        
        return True
    
    def generate_report(self, total_time):
        """生成下载完成报告"""
        print("\n" + "=" * 60)
        print("Visual Genome下载完成报告")
        print("=" * 60)
        
        print(f"总文件数: {self.stats['total_files']:,}")
        print(f"成功下载: {self.stats['downloaded']:,}")
        print(f"跳过文件: {self.stats['skipped']:,}")
        print(f"失败文件: {self.stats['failed']:,}")
        print(f"总下载大小: {self.stats['total_size'] / (1024**3):.2f} GB")
        print(f"总用时: {total_time / 3600:.2f} 小时")
        
        if self.stats['downloaded'] > 0:
            avg_speed = self.stats['total_size'] / total_time / (1024**2)
            print(f"平均速度: {avg_speed:.2f} MB/s")
        
        # 保存报告到文件
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": self.stats,
            "total_time_hours": total_time / 3600,
            "download_paths": {
                "VG_100K": str(self.vg_100k_dir),
                "VG_100K_2": str(self.vg_100k_2_dir)
            }
        }
        
        report_file = self.base_dir / "visual_genome_download_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {report_file}")
        print("=" * 60)

def main():
    """主函数"""
    print("Visual Genome数据集下载工具")
    print("此工具将下载完整的VG_100K和VG_100K_2图像集")
    
    # 确认下载
    response = input("\n确认开始下载？这将需要几个小时时间 (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("下载已取消")
        return
    
    # 创建下载器并开始下载
    downloader = VisualGenomeDownloader()
    success = downloader.download_all()
    
    if success:
        print("\n✅ Visual Genome数据集下载成功完成!")
    else:
        print("\n❌ 下载过程中遇到问题")

if __name__ == "__main__":
    main() 