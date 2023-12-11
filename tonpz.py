import numpy as np
import json

# 从JSON文件加载数据
with open('data.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 将JSON数据转换为NumPy数组
numpy_array = np.array(json_data)

# 保存为.npz文件
np.savez('data.npz', array=numpy_array)
