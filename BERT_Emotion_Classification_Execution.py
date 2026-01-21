# =====================================================
# 1. 基础库导入
# =====================================================
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import subprocess  # 用于快速获取大文件行数

# =====================================================
# 2. 设备设置
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# 3. 加载微调后的模型
# =====================================================
MODEL_DIR = "./goemotions_finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

if hasattr(model.config, "id2label"):
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
else:
    with open(os.path.join(MODEL_DIR, "labels.txt"), "r", encoding="utf-8") as f:
        label_names = [line.strip() for line in f]

num_labels = len(label_names)

# =====================================================
# 4. 预估总行数 (为了显示百分比进度条)
# =====================================================
INPUT_CSV = "split_song_lyrics_en_100k.csv"
OUTPUT_CSV = "split_song_lyrics_with_BERT2_emotions_en_100k_2.csv"

print("正在计算文件总行数以生成进度条...")


# 快速获取大文件行数的方法
def get_line_count(filename):
    lines = 0
    with open(filename, 'rb') as f:
        for line in f:
            lines += 1
    return lines - 1  # 减去表头行


total_lines = get_line_count(INPUT_CSV)
print(f"总计任务量: {total_lines} 行")

# =====================================================
# 5. 推理逻辑
# =====================================================
CHUNK_SIZE = 2000
BATCH_SIZE = 64

reader = pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)
first_chunk = True

# 关键修改：添加 total 参数和 bar_format 样式
with tqdm(
        total=total_lines,
        desc="Emotion Inference",
        unit="line",
        ncols=100,  # 进度条的总宽度
        ascii=False,  # 使用漂亮的方块填充
        colour='green'  # 进度条颜色（部分终端支持）
) as pbar:
    for chunk in reader:
        texts = chunk["lyric_line"].fillna("").astype(str).tolist()
        all_probs = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits)

            all_probs.append(probs.cpu().numpy())
            # 释放显存
            del inputs, outputs

        all_probs = np.vstack(all_probs)

        for idx, label in enumerate(label_names):
            chunk[label] = all_probs[:, idx]

        top_indices = np.argmax(all_probs, axis=1)
        chunk["top_emotion"] = [label_names[i] for i in top_indices]
        chunk["confidence"] = np.max(all_probs, axis=1)

        chunk.to_csv(
            OUTPUT_CSV,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig"
        )
        first_chunk = False

        # 更新进度条
        pbar.update(len(chunk))
        # 定期清理显存碎片
        torch.cuda.empty_cache()

print("\n===================================")
print("推理完成！")
print("输出文件：", OUTPUT_CSV)
print("===================================")