#!/usr/bin/env python3
"""
Visual Genome数据集测试下载脚本
先下载少量图像进行测试验证
"""

import os
import sys
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import json

class VGTestDownloader:
    def __init__(self, base_dir="D:/KKK/data/VAW/images"):
        self.base_dir = Path(base_dir)
        self.vg_100k_dir = self.base_dir / "VG_100K"
        
        # 测试配置：只下载前100个图像
        self.test_range = list(range(1, 101))
        self.base_url = "https://cs.stanford.edu/people/rak248/VG_100K/"
        
        # 下载配置
        self.max_workers = 4
        self.timeout = 30
        self.max_retries = 3
        
        # 统计信息
        self.stats = {
            "total": len(self.test_range),
            "downloaded": 0,
            "failed": 0,
            "skipped": 0
        }
        
        # 创建目录
        self.vg_100k_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"VG测试下载器初始化完成")
        print(f"测试范围: 图像1-100")
        print(f"目标目录: {self.vg_100k_dir}")
    
    def download_single_image(self, image_id):
        """下载单个图像文件"""
        filename = f"{image_id}.jpg"
        local_path = self.vg_100k_dir / filename
        
        # 检查文件是否已存在
        if local_path.exists() and local_path.stat().st_size > 0:
            self.stats["skipped"] += 1
            return {"status": "skipped", "file": filename}
        
        # 构建下载URL
        image_url = urljoin(self.base_url, filename)
        
        # 尝试下载
        for attempt in range(self.max_retries):
            try:
                print(f"正在下载: {filename} (尝试 {attempt + 1}/{self.max_retries})")
                
                response = requests.get(
                    image_url, 
                    timeout=self.timeout,
                    stream=True,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                
                if response.status_code == 200:
                    # 下载文件
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    file_size = local_path.stat().st_size
                    self.stats["downloaded"] += 1
                    
                    print(f"✅ 成功下载: {filename} ({file_size} bytes)")
                    return {"status": "success", "file": filename, "size": file_size}
                
                elif response.status_code == 404:
                    print(f"⚠️  文件不存在: {filename}")
                    self.stats["skipped"] += 1
                    return {"status": "not_found", "file": filename}
                
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 下载失败: {filename} - {str(e)}")
                if attempt == self.max_retries - 1:
                    self.stats["failed"] += 1
                    return {"status": "failed", "file": filename, "error": str(e)}
                else:
                    time.sleep(2)  # 等待重试
        
        return {"status": "failed", "file": filename}
    
    def test_download(self):
        """执行测试下载"""
        print("=" * 50)
        print("Visual Genome 测试下载开始")
        print("=" * 50)
        
        start_time = time.time()
        
        # 单线程下载（便于观察）
        for image_id in self.test_range:
            result = self.download_single_image(image_id)
            
            # 显示进度
            completed = self.stats["downloaded"] + self.stats["failed"] + self.stats["skipped"]
            print(f"进度: {completed}/{self.stats['total']} "
                  f"(成功:{self.stats['downloaded']}, 失败:{self.stats['failed']}, 跳过:{self.stats['skipped']})")
            
            # 每10个文件显示一次状态
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                print(f"已用时: {elapsed:.1f}秒")
        
        # 生成测试报告
        total_time = time.time() - start_time
        self.generate_test_report(total_time)
        
        return self.stats["downloaded"] > 0
    
    def generate_test_report(self, total_time):
        """生成测试报告"""
        print("\n" + "=" * 50)
        print("Visual Genome 测试下载报告")
        print("=" * 50)
        
        print(f"测试范围: 图像1-100")
        print(f"成功下载: {self.stats['downloaded']}")
        print(f"跳过文件: {self.stats['skipped']}")
        print(f"失败文件: {self.stats['failed']}")
        print(f"总用时: {total_time:.2f} 秒")
        
        if self.stats['downloaded'] > 0:
            avg_time = total_time / self.stats['downloaded']
            print(f"平均下载时间: {avg_time:.2f} 秒/文件")
        
        # 检查下载的文件
        downloaded_files = list(self.vg_100k_dir.glob("*.jpg"))
        print(f"实际下载文件数: {len(downloaded_files)}")
        
        if downloaded_files:
            total_size = sum(f.stat().st_size for f in downloaded_files)
            print(f"总下载大小: {total_size / (1024*1024):.2f} MB")
            print(f"平均文件大小: {total_size / len(downloaded_files) / 1024:.1f} KB")
        
        print("=" * 50)
        
        # 评估是否可以继续完整下载
        if self.stats['downloaded'] >= 10:
            print("✅ 测试成功！网络连接和下载功能正常")
            print("💡 建议：可以继续进行完整数据集下载")
        else:
            print("⚠️  测试结果不理想，建议检查网络连接")

def main():
    """主函数"""
    print("Visual Genome 测试下载工具")
    print("此工具将下载前100个图像进行测试")
    
    # 创建下载器并开始测试
    downloader = VGTestDownloader()
    success = downloader.test_download()
    
    if success:
        print("\n🎉 测试下载完成!")
        response = input("\n是否继续完整下载？(y/N): ")
        if response.lower() in ['y', 'yes']:
            print("请运行 python download_visual_genome.py 进行完整下载")
    else:
        print("\n❌ 测试下载失败")

if __name__ == "__main__":
    main() 