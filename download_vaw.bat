@echo off
echo 开始下载VAW数据集...

:: 创建目录
if not exist "D:\KKK\data\VAW\annotations" mkdir "D:\KKK\data\VAW\annotations"
if not exist "D:\KKK\data\VAW\images" mkdir "D:\KKK\data\VAW\images"

:: 下载VAW标注文件
echo 下载VAW标注文件...
curl -L -o "D:\KKK\data\VAW\annotations\train_part1.json" "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part1.json"
curl -L -o "D:\KKK\data\VAW\annotations\train_part2.json" "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part2.json"
curl -L -o "D:\KKK\data\VAW\annotations\val.json" "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/val.json"
curl -L -o "D:\KKK\data\VAW\annotations\test.json" "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/test.json"

echo VAW标注文件下载完成
echo 检查文件大小...
dir "D:\KKK\data\VAW\annotations\*.json"

echo VAW数据集下载完成！
pause 