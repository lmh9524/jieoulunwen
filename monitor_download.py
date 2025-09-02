#!/usr/bin/env python3
"""
Visual Genome下载监控脚本
实时监控下载进度和状态
"""

import os
import time
import json
from pathlib import Path
import psutil

class DownloadMonitor:
    def __init__(self, base_dir="D:/KKK/data/VAW/images"):
        self.base_dir = Path(base_dir)
        self.vg_100k_dir = self.base_dir / "VG_100K"
        self.vg_100k_2_dir = self.base_dir / "VG_100K_2"
        self.report_file = self.base_dir / "visual_genome_download_report.json"
        
        # 预期文件数量
        self.expected_vg_100k = 50000
        self.expected_vg_100k_2 = 58000
        self.total_expected = self.expected_vg_100k + self.expected_vg_100k_2
        
        print("Visual Genome下载监控器启动")
        print(f"监控目录: {self.base_dir}")
    
    def count_files(self, directory):
        """统计目录中的文件数量和大小"""
        if not directory.exists():
            return 0, 0
        
        jpg_files = list(directory.glob("*.jpg"))
        count = len(jpg_files)
        total_size = sum(f.stat().st_size for f in jpg_files if f.exists())
        
        return count, total_size
    
    def check_download_process(self):
        """检查是否有下载进程在运行"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('download_visual_genome.py' in arg for arg in cmdline):
                    return True, proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False, None
    
    def get_download_stats(self):
        """获取当前下载统计"""
        vg_100k_count, vg_100k_size = self.count_files(self.vg_100k_dir)
        vg_100k_2_count, vg_100k_2_size = self.count_files(self.vg_100k_2_dir)
        
        total_count = vg_100k_count + vg_100k_2_count
        total_size = vg_100k_size + vg_100k_2_size
        
        # 计算进度百分比
        progress_percent = (total_count / self.total_expected) * 100 if self.total_expected > 0 else 0
        
        return {
            "vg_100k": {"count": vg_100k_count, "size": vg_100k_size},
            "vg_100k_2": {"count": vg_100k_2_count, "size": vg_100k_2_size},
            "total": {"count": total_count, "size": total_size},
            "progress_percent": progress_percent
        }
    
    def format_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.2f} GB"
    
    def display_status(self, stats, is_running, pid):
        """显示当前状态"""
        os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
        
        print("=" * 60)
        print("Visual Genome 数据集下载监控")
        print("=" * 60)
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if is_running:
            print(f"状态: 🟢 下载进行中 (PID: {pid})")
        else:
            print("状态: 🔴 下载进程未运行")
        
        print()
        print("📊 下载进度:")
        print(f"VG_100K:   {stats['vg_100k']['count']:,} / {self.expected_vg_100k:,} 文件 "
              f"({self.format_size(stats['vg_100k']['size'])})")
        print(f"VG_100K_2: {stats['vg_100k_2']['count']:,} / {self.expected_vg_100k_2:,} 文件 "
              f"({self.format_size(stats['vg_100k_2']['size'])})")
        print(f"总计:      {stats['total']['count']:,} / {self.total_expected:,} 文件 "
              f"({self.format_size(stats['total']['size'])})")
        
        # 进度条
        progress = stats['progress_percent']
        bar_length = 40
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"进度:      [{bar}] {progress:.1f}%")
        
        # 估算剩余时间（如果有历史数据）
        if hasattr(self, 'last_stats') and hasattr(self, 'last_time'):
            time_diff = time.time() - self.last_time
            count_diff = stats['total']['count'] - self.last_stats['total']['count']
            
            if count_diff > 0 and time_diff > 0:
                rate = count_diff / time_diff  # 文件/秒
                remaining_files = self.total_expected - stats['total']['count']
                eta_seconds = remaining_files / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600
                
                print(f"下载速度:  {rate:.2f} 文件/秒")
                print(f"预计剩余:  {eta_hours:.1f} 小时")
        
        print()
        print("💡 提示:")
        print("- 按 Ctrl+C 退出监控")
        print("- 下载进程在后台运行，关闭监控不影响下载")
        
        # 检查是否有下载报告
        if self.report_file.exists():
            print("- 发现下载完成报告，下载可能已完成")
        
        print("=" * 60)
    
    def monitor(self, refresh_interval=10):
        """开始监控"""
        print(f"开始监控，每{refresh_interval}秒刷新一次...")
        
        try:
            while True:
                # 获取当前状态
                stats = self.get_download_stats()
                is_running, pid = self.check_download_process()
                
                # 显示状态
                self.display_status(stats, is_running, pid)
                
                # 保存当前状态用于计算速度
                self.last_stats = stats
                self.last_time = time.time()
                
                # 检查是否完成
                if stats['progress_percent'] >= 99.0:
                    print("\n🎉 下载接近完成！")
                    if not is_running:
                        print("✅ 下载进程已结束，可能已完成下载")
                        break
                
                # 等待下次刷新
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            print("下载进程仍在后台运行...")

def main():
    """主函数"""
    monitor = DownloadMonitor()
    monitor.monitor()

if __name__ == "__main__":
    main() 