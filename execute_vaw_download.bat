@echo off
echo [执行模式] 开始VAW数据集重新下载...

echo 步骤1: 创建目录结构
if not exist "D:\KKK\data\VAW\annotations" mkdir "D:\KKK\data\VAW\annotations"
if not exist "D:\KKK\data\VAW\images" mkdir "D:\KKK\data\VAW\images"
if not exist "D:\KKK\data\VAW\metadata" mkdir "D:\KKK\data\VAW\metadata"
echo 目录创建完成

echo.
echo 步骤2: 下载VAW标注文件
echo 下载 train_part1.json...
curl -L -# -o "D:\KKK\data\VAW\annotations\train_part1.json" "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part1.json"

echo 下载 train_part2.json...
curl -L -# -o "D:\KKK\data\VAW\annotations\train_part2.json" "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part2.json"

echo 下载 val.json...
curl -L -# -o "D:\KKK\data\VAW\annotations\val.json" "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/val.json"

echo 下载 test.json...
curl -L -# -o "D:\KKK\data\VAW\annotations\test.json" "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/test.json"

echo.
echo 步骤3: 验证下载结果
echo 检查文件大小：
dir "D:\KKK\data\VAW\annotations\*.json"

echo.
echo 步骤4: 下载样本Visual Genome图像
echo 下载前10张图像作为测试...
curl -L -# -o "D:\KKK\data\VAW\images\1.jpg" "https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg"
curl -L -# -o "D:\KKK\data\VAW\images\2.jpg" "https://cs.stanford.edu/people/rak248/VG_100K_2/2.jpg"
curl -L -# -o "D:\KKK\data\VAW\images\3.jpg" "https://cs.stanford.edu/people/rak248/VG_100K/3.jpg"
curl -L -# -o "D:\KKK\data\VAW\images\4.jpg" "https://cs.stanford.edu/people/rak248/VG_100K/4.jpg"
curl -L -# -o "D:\KKK\data\VAW\images\5.jpg" "https://cs.stanford.edu/people/rak248/VG_100K_2/5.jpg"

echo.
echo 检查图像文件：
dir "D:\KKK\data\VAW\images\*.jpg"

echo.
echo VAW数据集重新下载执行完成！
echo 请运行验证脚本检查数据集完整性。 