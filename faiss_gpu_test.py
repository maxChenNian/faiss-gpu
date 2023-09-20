import faiss
import numpy as np
import time

# 准备一些示例向量数据
# 这里使用随机生成的数据作为示例
dimension = 8000
num_vectors = 100000
query_vector = np.random.rand(dimension).astype('float32')  # 查询向量
data = np.random.rand(num_vectors, dimension).astype('float32')  # 数据集

# 创建Flat索引（在GPU上）
index = faiss.IndexFlatL2(dimension)  # L2距离度量

# 将数据添加到GPU索引中
index = faiss.index_cpu_to_all_gpus(index)  # 将CPU索引移动到GPU

# 添加数据到索引
start = time.time()
index.add(data)
print("数据库载入时间：", time.time() - start)

# 设置要返回的最近邻居数量
k = 5

# 在GPU上进行相似性搜索
start = time.time()
distances, indices = index.search(np.array([query_vector]), k)
print("GPU搜索时间：", time.time() - start)
# 打印查询结果
print("最近邻居的索引：", indices)
print("对应的距离：", distances)

for i in range(5):
    start = time.time()
    distances, indices = index.search(np.array([query_vector]), k)
    print(i, "GPU搜索时间：", time.time() - start)
    # 打印查询结果
    print("  最近邻居的索引：", indices)
    print("  对应的距离：", distances)
