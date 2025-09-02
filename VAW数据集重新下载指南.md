# VAW数据集重新下载指南

## 📋 当前问题分析

目前VAW数据集状态：
- ✅ 目录结构已创建
- ⚠️ 标注文件部分损坏（train_part2.json）
- ❌ 图像文件缺失（VAW使用Visual Genome图像）

## 🎯 重新下载方案

### 方案1：手动浏览器下载（推荐）

#### 1.1 下载VAW标注文件
访问以下链接，右键"另存为"到 `D:\KKK\data\VAW\annotations\` 目录：

1. **train_part1.json**:
   - 链接：https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part1.json
   - 预期大小：~85MB

2. **train_part2.json**:
   - 链接：https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part2.json
   - 预期大小：~7MB

3. **val.json**:
   - 链接：https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/val.json
   - 预期大小：~15MB

4. **test.json**:
   - 链接：https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/test.json
   - 预期大小：~28MB

#### 1.2 下载Visual Genome图像样本
由于完整的Visual Genome图像集很大（~15GB），建议先下载少量样本进行测试：

**方法A - 通过Hugging Face（推荐）**:
1. 访问：https://huggingface.co/datasets/ranjaykrishna/visual_genome
2. 下载部分图像文件到 `D:\KKK\data\VAW\images\`

**方法B - 直接下载**:
从Stanford服务器下载部分图像：
- 基础URL1：https://cs.stanford.edu/people/rak248/VG_100K/
- 基础URL2：https://cs.stanford.edu/people/rak248/VG_100K_2/
- 图像命名格式：`{image_id}.jpg`

### 方案2：PowerShell命令行下载

打开PowerShell，执行以下命令：

```powershell
# 创建目录
New-Item -ItemType Directory -Path "D:\KKK\data\VAW\annotations" -Force
New-Item -ItemType Directory -Path "D:\KKK\data\VAW\images" -Force

# 下载VAW标注文件
$baseUrl = "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data"
$files = @("train_part1.json", "train_part2.json", "val.json", "test.json")

foreach($file in $files) {
    $url = "$baseUrl/$file"
    $dest = "D:\KKK\data\VAW\annotations\$file"
    Write-Host "下载: $file"
    Invoke-WebRequest -Uri $url -OutFile $dest -Headers @{"User-Agent"="Mozilla/5.0"}
    Write-Host "完成: $file"
}

# 检查下载结果
Get-ChildItem "D:\KKK\data\VAW\annotations" | Select-Object Name, Length
```

### 方案3：Python脚本下载

如果Python环境正常，使用我创建的 `download_vaw_complete.py` 脚本：

```bash
python download_vaw_complete.py
```

## 🔍 下载后验证

下载完成后，请验证文件：

### 1. 检查标注文件大小
- train_part1.json: ~85MB
- train_part2.json: ~7MB  
- val.json: ~15MB
- test.json: ~28MB

### 2. 验证JSON格式
可以用文本编辑器打开文件，确认是有效的JSON格式，而不是HTML错误页面。

### 3. 运行验证脚本
下载完成后运行：
```bash
python check_dataset_completeness.py
```

## 📊 预期结果

完整下载后的目录结构：
```
D:\KKK\data\VAW\
├── annotations\
│   ├── train_part1.json        ✅ ~85MB, 108,395条记录
│   ├── train_part2.json        ✅ ~7MB, 完整JSON
│   ├── val.json                ✅ ~15MB, 验证集
│   └── test.json               ✅ ~28MB, 31,819条记录
├── images\                     ✅ 部分Visual Genome图像
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── metadata\                   ✅ 已创建
    └── dataset_summary.json    📊 数据集摘要
```

## 🚀 后续步骤

1. **优先使用方案1**（浏览器手动下载）确保文件完整性
2. 下载完成后运行验证脚本确认数据集状态
3. 如需完整的Visual Genome图像，可以后续分批下载
4. 验证通过后即可开始VAW相关的实验

## ⚠️ 注意事项

- VAW数据集总共涉及72,274张图像，建议先下载1000张进行测试
- 完整的Visual Genome图像集约15GB，请确保有足够磁盘空间
- 如网络不稳定，可以分批多次下载

---
**状态更新**：请选择适合的方案重新下载VAW数据集，推荐从方案1开始。 