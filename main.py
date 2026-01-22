from annoy import AnnoyIndex
import pickle
f = 28
u = AnnoyIndex(f, 'angular')
u.load('songs_emotion_vector_index.ann')
with open('id_to_title.pkl', 'rb') as f:
    id_to_title = pickle.load(f)

# 假设用户当前的情感向量（28 维）
user_vector = [0.372, 0.845, 0.129, 0.663, 0.514, 0.927, 0.238, 0.781,
 0.459, 0.606, 0.194, 0.883, 0.051, 0.734, 0.368, 0.592,
 0.809, 0.147, 0.425, 0.971, 0.286, 0.654, 0.902, 0.113,
 0.538, 0.779, 0.047, 0.691]


# 搜索最相近的 5 首歌
# include_distances=True 会返回相似度得分（Annoy 返回的是经过转换的距离）
indices, distances = u.get_nns_by_vector(user_vector, 5, include_distances=True)

# 输出结果
for idx, dist in zip(indices, distances):
    print(f"おすすめの曲: {id_to_title[idx]},COS類似度: {dist}")