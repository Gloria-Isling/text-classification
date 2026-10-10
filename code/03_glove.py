import os
import re
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "nyt_train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "nyt_val.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "nyt_test.csv")

GLOVE_PATH = os.path.join(
    BASE_DIR,
    "glove",
    "glove.6B.100d.txt"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

EMBEDDING_DIM = 100



print("=" * 70)
print("Task 2 - GloVe 100d + Logistic Regression")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\n数据读取完成：")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))



print("\n正在加载 GloVe 100d...")
print("文件：", GLOVE_PATH)

glove = {}

with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading GloVe"):
        values = line.rstrip().split(" ")

        word = values[0]

        vector = np.asarray(
            values[1:],
            dtype=np.float32
        )

        # 只接受正确的 100 维向量
        if len(vector) == EMBEDDING_DIM:
            glove[word] = vector

print("\nGloVe 加载完成！")
print("词向量数量：", len(glove))
print("向量维度：", EMBEDDING_DIM)



def tokenize(text):
    """
    简单英文 tokenizer：
    转小写并提取英文单词。
    """

    text = str(text).lower()

    return re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        text
    )



def text_to_glove(text):

    words = tokenize(text)

    vectors = []

    for word in words:
        if word in glove:
            vectors.append(glove[word])

    # 如果整篇文章没有任何词出现在 GloVe 中
    if len(vectors) == 0:
        return np.zeros(
            EMBEDDING_DIM,
            dtype=np.float32
        )

    return np.mean(
        vectors,
        axis=0
    ).astype(np.float32)



def create_glove_features(texts, name):

    features = []

    print(f"\n正在生成 {name} GloVe 特征...")

    for text in tqdm(
        texts,
        desc=name
    ):
        features.append(
            text_to_glove(text)
        )

    return np.vstack(features)


X_train = create_glove_features(
    train_df["text"],
    "Training Set"
)

X_val = create_glove_features(
    val_df["text"],
    "Validation Set"
)

X_test = create_glove_features(
    test_df["text"],
    "Test Set"
)

y_train = train_df["label"].astype(str)
y_val = val_df["label"].astype(str)
y_test = test_df["label"].astype(str)

print("\n特征生成完成：")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)



print("\n" + "=" * 70)
print("训练 Logistic Regression")
print("=" * 70)

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\n模型训练完成！")




val_pred = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

val_macro_f1 = f1_score(
    y_val,
    val_pred,
    average="macro"
)

print("\n" + "=" * 70)
print("Validation Set")
print("=" * 70)

print(
    f"Accuracy : {val_accuracy:.6f}"
)

print(
    f"Macro-F1 : {val_macro_f1:.6f}"
)



test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_macro_f1 = f1_score(
    y_test,
    test_pred,
    average="macro"
)

print("\n" + "=" * 70)
print("Test Set")
print("=" * 70)

print(
    f"Accuracy : {test_accuracy:.6f}"
)

print(
    f"Macro-F1 : {test_macro_f1:.6f}"
)

print("\nTest Set Classification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        digits=4
    )
)



result = {
    "method": "GloVe 100d",
    "embedding_dimension": EMBEDDING_DIM,
    "classifier": "Logistic Regression",

    "validation_accuracy":
        float(val_accuracy),

    "validation_macro_f1":
        float(val_macro_f1),

    "test_accuracy":
        float(test_accuracy),

    "test_macro_f1":
        float(test_macro_f1)
}

RESULT_PATH = os.path.join(
    RESULTS_DIR,
    "glove.json"
)

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




print("\n" + "=" * 70)
print("GloVe 最终结果")
print("=" * 70)

print(
    f"Validation Accuracy : {val_accuracy:.6f}"
)

print(
    f"Validation Macro-F1 : {val_macro_f1:.6f}"
)

print(
    f"Test Accuracy       : {test_accuracy:.6f}"
)

print(
    f"Test Macro-F1       : {test_macro_f1:.6f}"
)

print("\n结果已保存：")
print(RESULT_PATH)

print("\n03_glove.py 执行完成！")