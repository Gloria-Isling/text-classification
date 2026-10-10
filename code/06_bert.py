import os
import json
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm



SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models", "bert")

TRAIN_PATH = os.path.join(DATA_DIR, "nyt_train.csv")
VAL_PATH = os.path.join(DATA_DIR, "nyt_val.csv")
TEST_PATH = os.path.join(DATA_DIR, "nyt_test.csv")

RESULT_PATH = os.path.join(RESULT_DIR, "bert.json")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)



MODEL_NAME = "google-bert/bert-base-uncased"

MAX_LENGTH = 64
EPOCHS = 3

# RTX 3050 Ti Laptop GPU 使用 16 比较稳妥
BATCH_SIZE = 16

LEARNING_RATE = 2e-5



print("=" * 80)
print("Task 3 - BERT Fine-tuning")
print("=" * 80)

print("\n作业指定参数：")
print(f"Pre-trained Model : {MODEL_NAME}")
print(f"Maximum Length    : {MAX_LENGTH}")
print(f"Epochs            : {EPOCHS}")
print(f"Batch Size        : {BATCH_SIZE}")
print(f"Learning Rate     : {LEARNING_RATE}")



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("\nCUDA 可用！")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    DEVICE = torch.device("cpu")
    print("\n警告：CUDA 不可用，将使用 CPU。")


print("当前设备:", DEVICE)



print("\n" + "=" * 80)
print("读取 NYT 数据集")
print("=" * 80)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)


# 删除缺失值
train_df = train_df.dropna(subset=["text", "label"]).reset_index(drop=True)
val_df = val_df.dropna(subset=["text", "label"]).reset_index(drop=True)
test_df = test_df.dropna(subset=["text", "label"]).reset_index(drop=True)


print("\n数据读取完成：")
print("Training Set  :", len(train_df))
print("Validation Set:", len(val_df))
print("Test Set      :", len(test_df))


print("\nTraining Set 类别分布：")
print(train_df["label"].value_counts())



# 为保证固定顺序，明确指定三个类别
LABELS = ["business", "politics", "sports"]

label2id = {
    label: idx
    for idx, label in enumerate(LABELS)
}

id2label = {
    idx: label
    for label, idx in label2id.items()
}


print("\nLabel Mapping:")
print(label2id)


def encode_labels(df):
    encoded = df["label"].map(label2id)

    if encoded.isnull().any():
        unknown = df.loc[encoded.isnull(), "label"].unique()
        raise ValueError(
            f"发现未知类别：{unknown}"
        )

    return encoded.astype(int).tolist()


train_labels = encode_labels(train_df)
val_labels = encode_labels(val_df)
test_labels = encode_labels(test_df)



print("\n" + "=" * 80)
print("加载 BERT Tokenizer")
print("=" * 80)

print("\n第一次运行时会自动从 Hugging Face 下载模型文件。")
print("Model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("\nTokenizer 加载完成！")



class NYTDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long
                )
        }

        return item



print("\n" + "=" * 80)
print("创建 BERT Dataset")
print("=" * 80)

train_dataset = NYTDataset(
    train_df["text"].astype(str).tolist(),
    train_labels,
    tokenizer,
    MAX_LENGTH
)

val_dataset = NYTDataset(
    val_df["text"].astype(str).tolist(),
    val_labels,
    tokenizer,
    MAX_LENGTH
)

test_dataset = NYTDataset(
    test_df["text"].astype(str).tolist(),
    test_labels,
    tokenizer,
    MAX_LENGTH
)


print("\nDataset 创建完成：")
print("Train      :", len(train_dataset))
print("Validation :", len(val_dataset))
print("Test       :", len(test_dataset))



train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\nDataLoader 创建完成。")



print("\n" + "=" * 80)
print("加载 BERT Classification Model")
print("=" * 80)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

model.to(DEVICE)

print("\n模型加载完成！")
print("模型已移动到:", DEVICE)



optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)



def evaluate_model(model, data_loader, description):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    progress_bar = tqdm(
        data_loader,
        desc=description
    )

    with torch.no_grad():

        for batch in progress_bar:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    avg_loss = total_loss / len(data_loader)

    return (
        avg_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )



print("\n" + "=" * 80)
print("开始 BERT Fine-tuning")
print("=" * 80)

print("\n训练要求：")
print("Epochs     :", EPOCHS)
print("Max Length :", MAX_LENGTH)
print("Device     :", DEVICE)


training_history = []


for epoch in range(EPOCHS):

    print("\n" + "#" * 80)
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print("#" * 80)

    model.train()

    total_train_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch + 1}"
    )

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    avg_train_loss = (
        total_train_loss / len(train_loader)
    )



    (
        val_loss,
        val_accuracy,
        val_macro_f1,
        _,
        _
    ) = evaluate_model(
        model,
        val_loader,
        f"Validation Epoch {epoch + 1}"
    )


    print("\nEpoch", epoch + 1, "结果")
    print("-" * 60)

    print(
        f"Training Loss     : "
        f"{avg_train_loss:.6f}"
    )

    print(
        f"Validation Loss   : "
        f"{val_loss:.6f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.6f}"
    )

    print(
        f"Validation Macro-F1: "
        f"{val_macro_f1:.6f}"
    )


    training_history.append(
        {
            "epoch": epoch + 1,
            "train_loss":
                float(avg_train_loss),

            "validation_loss":
                float(val_loss),

            "validation_accuracy":
                float(val_accuracy),

            "validation_macro_f1":
                float(val_macro_f1)
        }
    )



print("\n" + "=" * 80)
print("Final Validation Set Evaluation")
print("=" * 80)

(
    final_val_loss,
    final_val_accuracy,
    final_val_macro_f1,
    _,
    _
) = evaluate_model(
    model,
    val_loader,
    "Final Validation"
)


print("\nValidation Set")

print(
    f"Accuracy : "
    f"{final_val_accuracy:.6f}"
)

print(
    f"Macro-F1 : "
    f"{final_val_macro_f1:.6f}"
)



print("\n" + "=" * 80)
print("NYT Test Set Evaluation")
print("=" * 80)

(
    test_loss,
    test_accuracy,
    test_macro_f1,
    test_true,
    test_pred
) = evaluate_model(
    model,
    test_loader,
    "Testing"
)


print("\nNYT Test Set")

print(
    f"Accuracy : "
    f"{test_accuracy:.6f}"
)

print(
    f"Macro-F1 : "
    f"{test_macro_f1:.6f}"
)



print("\nNYT Test Set Classification Report:\n")

print(
    classification_report(
        test_true,
        test_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4
    )
)



print("\n" + "=" * 80)
print("保存 BERT 模型")
print("=" * 80)

model.save_pretrained(
    MODEL_DIR
)

tokenizer.save_pretrained(
    MODEL_DIR
)

print("\n模型保存到：")
print(MODEL_DIR)



result = {

    "method":
        "BERT Fine-tuning",

    "model":
        MODEL_NAME,

    "max_length":
        MAX_LENGTH,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "vector_model":
        "BERT-base-uncased",

    "training_history":
        training_history,

    "validation_accuracy":
        float(final_val_accuracy),

    "validation_macro_f1":
        float(final_val_macro_f1),

    "test_accuracy":
        float(test_accuracy),

    "test_macro_f1":
        float(test_macro_f1)
}


with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4,
        ensure_ascii=False
    )



print("\n" + "=" * 80)
print("BERT Fine-tuning 最终结果")
print("=" * 80)

print(
    f"Pre-trained Model      : "
    f"{MODEL_NAME}"
)

print(
    f"Maximum Sequence Length: "
    f"{MAX_LENGTH}"
)

print(
    f"Number of Epochs       : "
    f"{EPOCHS}"
)

print(
    f"Validation Accuracy    : "
    f"{final_val_accuracy:.6f}"
)

print(
    f"Validation Macro-F1    : "
    f"{final_val_macro_f1:.6f}"
)

print(
    f"NYT Test Accuracy      : "
    f"{test_accuracy:.6f}"
)

print(
    f"NYT Test Macro-F1      : "
    f"{test_macro_f1:.6f}"
)


print("\n结果已保存：")
print(RESULT_PATH)

print("\n06_bert.py 执行完成！")