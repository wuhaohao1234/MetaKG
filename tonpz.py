import numpy as np
import json

# 从JSON文件加载数据
with open('data.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 将JSON数据转换为NumPy数组
numpy_array = np.array(json_data)

# 保存为.npz文件
np.savez('data.npz', array=numpy_array)

# import chardet
# import csv
# import numpy as np

# # 检测编码
# with open('MetaKG/datasets/citeulike-a-master/raw-data.csv', 'rb') as f:
#     result = chardet.detect(f.read())

# # 使用检测到的编码加载数据
# with open('MetaKG/datasets/citeulike-a-master/raw-data.csv', 'r', encoding=result['encoding']) as f:
#     # 使用csv.reader解析CSV文件
#     csv_reader = csv.reader(f, delimiter=',')
    
#     # 读取头部（列名）
#     header = next(csv_reader)
    
#     # 读取数据
#     data = np.genfromtxt(f, delimiter=',', dtype=None, names=True, missing_values='', filling_values=np.nan)

# # 打印数据
# print(data)

# # 将数据保存到 NPZ 文件
# np.savez('MetaKG/data.npz', data=data)
