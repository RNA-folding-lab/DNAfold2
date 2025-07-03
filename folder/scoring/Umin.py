import numpy as np

# 读取文件
data = np.loadtxt('Energy_0.dat')  # 默认以空格或制表符分隔

# 获取第二列数据和对应的行号
second_column = data[:, 1]
indices = np.argsort(second_column)  # 排序并获取索引

# 确定要取的数量（不超过500）
n = min(500, len(indices))
top_indices = indices[:n]  # 取前n个最小值的行号

# 保存结果到文件
np.savetxt('min.dat', top_indices + 1, fmt='%d')  # 行号从1开始而非0

