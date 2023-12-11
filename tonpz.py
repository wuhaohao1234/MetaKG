import chardet
import csv
import numpy as np

# 检测编码
with open('MetaKG/datasets/citeulike-a-master/raw-data.csv', 'rb') as f:
    result = chardet.detect(f.read())

# 使用检测到的编码加载数据
with open('MetaKG/datasets/citeulike-a-master/raw-data.csv', 'r', encoding=result['encoding']) as f:
    # 使用csv.reader解析CSV文件
    csv_reader = csv.reader(f, delimiter=',')
    
    # 读取头部（列名）
    header = next(csv_reader)
    
    # 读取数据
    data = np.genfromtxt(f, delimiter=',', dtype=None, names=True, missing_values='', filling_values=np.nan)

# 打印数据
print(data)

# 将数据保存到 NPZ 文件
np.savez('MetaKG/data.npz', data=data)
