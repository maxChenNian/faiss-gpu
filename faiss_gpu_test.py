import faiss
import numpy as np

# 准备一些示例向量数据
# 你可以替换这里的数据为你自己的向量数据
data = np.random.rand(100, 128).astype('float32')  # 100个128维的随机向量

# 创建Flat索引
index = faiss.IndexFlatL2(128)  # L2距离度量
# 将数据添加到索引中
index.add(data)

# 假设你的查询向量是一个128维的随机向量
query_vector = np.random.rand(128).astype('float32')

# 设置要返回的最近邻居数量
k = 5

# 进行相似性搜索
distances, indices = index.search(np.array([query_vector]), k)

# 打印查询结果
print("最近邻居的索引：", indices)
print("对应的距离：", distances)
