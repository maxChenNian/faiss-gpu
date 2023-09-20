import faiss
import numpy as np
import time

dimension = 128
num_vectors = 10000
query_vector = np.random.rand(dimension).astype('float32')
data = np.random.rand(num_vectors, dimension).astype('float32')

index = faiss.IndexFlatL2(dimension)

# index = faiss.index_cpu_to_all_gpus(index)
print("nihao")
index.add(data)

k = 5

start = time.time()
distances, indices = index.search(np.array([query_vector]), k)
print("gpu time:", time.time() - start)
print("index:", indices)
print("dis:", distances)
