
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch # 确保导入了 torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

# =====================================================
# 1. 基础配置
# =====================================================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型名（微调通常使用基础 BERT/RoBERTa 变体）
model_name = "sentence-transformers/all-mpnet-base-v2"
# 或者使用 "bert-base-uncased" 等

# =====================================================
# 2. 加载数据集与标签
# =====================================================
dataset = load_dataset("mrm8488/goemotions")
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
num_labels = len(emotion_labels)

# =====================================================
# 3. 初始化 Tokenizer 和模型 (端到端)
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 使用 AutoModel 自动构建：Transformer 主体 + 线性分类层
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"  # 关键：指定多标签任务
).to(device)


# =====================================================
# 4. 微调专用的 Dataset
# =====================================================
class GoEmotionsFTDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, emotion_labels, max_len=128):
        self.texts = hf_dataset["text"]
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 转换标签矩阵 (N, 28)
        labels_list = []
        for label in emotion_labels:
            labels_list.append(hf_dataset[label])
        self.labels = np.array(labels_list).T.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }


# 切分并构建 Loader
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = GoEmotionsFTDataset(split_dataset["train"], tokenizer, emotion_labels)
val_dataset = GoEmotionsFTDataset(split_dataset["test"], tokenizer, emotion_labels)

# 微调显存消耗大，batch_size 建议设为 16 或 32
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# =====================================================
# 5. 优化器与学习率调度
# =====================================================
# 微调通常需要非常小的学习率 (如 2e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)~

EPOCHS = 3  # 微调通常 3-5 个 epoch 即可
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# =====================================================
# 6. 训练与验证函数
# =====================================================
def train_fn(model, dataloader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_fn(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            outputs = model(input_ids, attention_mask=attention_mask)
            # 对于多标签，使用 Sigmoid + 阈值
            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (preds >= 0.5).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    return {
        "micro": f1_score(all_labels, all_preds, average="micro"),
        "macro": f1_score(all_labels, all_preds, average="macro")
    }


# =====================================================
# 7. 主训练循环
# =====================================================
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1} ---")
    avg_loss = train_fn(model, train_loader)
    metrics = eval_fn(model, val_loader)

    print(f"Loss: {avg_loss:.4f} | Micro-F1: {metrics['micro']:.4f} | Macro-F1: {metrics['macro']:.4f}")

# =====================================================
# 8. 保存微调后的模型
# =====================================================
save_path = "./goemotions_finetuned_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"模型已微调并保存至: {save_path}")