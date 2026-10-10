import os
import re
import json
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

NYT_TRAIN_PATH = os.path.join(DATA_DIR, "nyt_train.csv")
NYT_VAL_PATH = os.path.join(DATA_DIR, "nyt_val.csv")
NYT_TEST_PATH = os.path.join(DATA_DIR, "nyt_test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "word2vec_nyt.model")
RESULT_PATH = os.path.join(RESULT_DIR, "word2vec_nyt.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)



SEED = 42

random.seed(SEED)
np.random.seed(SEED)



VECTOR_SIZE = 100

WINDOW = 5
MIN_COUNT = 2
WORKERS = 4
EPOCHS = 10



def tokenize(text):
    """
    英文文本分词：
    1. 转小写
    2. 提取英文单词
    """

    text = str(text).lower()

    tokens = re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        text
    )

    return tokens



def load_nyt_data():

    print("=" * 80)
    print("读取 NYT 数据集")
    print("=" * 80)

    train_df = pd.read_csv(NYT_TRAIN_PATH)
    val_df = pd.read_csv(NYT_VAL_PATH)
    test_df = pd.read_csv(NYT_TEST_PATH)

    print()
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    # 检查必须存在的列
    required_columns = ["text", "label"]

    for name, df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]:

        for col in required_columns:

            if col not in df.columns:

                raise ValueError(
                    f"{name} 数据缺少列：{col}，"
                    f"当前列为：{list(df.columns)}"
                )

    print()
    print("NYT 数据读取完成！")

    return train_df, val_df, test_df



def train_word2vec(train_texts):

    print()
    print("=" * 80)
    print("训练 Word2Vec")
    print("=" * 80)

    print()
    print("训练数据：NYT Training Set")
    print("注意：Validation 和 Test 不参与 Word2Vec 训练")

    print()
    print("正在对 NYT Training Set 进行分词...")

    tokenized_train = []

    for text in tqdm(
        train_texts,
        desc="Tokenizing NYT Training Set"
    ):

        tokens = tokenize(text)

        if len(tokens) > 0:
            tokenized_train.append(tokens)

    print()
    print(
        "有效 NYT Training 文档数量:",
        len(tokenized_train)
    )

    print()
    print("Word2Vec 参数:")

    print(
        "vector_size =",
        VECTOR_SIZE
    )

    print(
        "window      =",
        WINDOW
    )

    print(
        "min_count   =",
        MIN_COUNT
    )

    print(
        "epochs      =",
        EPOCHS
    )

    print()
    print(
        "开始使用 NYT Training Set 训练 Word2Vec..."
    )

    model = Word2Vec(
        sentences=tokenized_train,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
        seed=SEED
    )

    print()
    print("Word2Vec 训练完成！")

    print(
        "词向量维度:",
        model.vector_size
    )

    print(
        "词表大小:",
        len(model.wv)
    )

    # 保存 Word2Vec 模型
    model.save(MODEL_PATH)

    print()
    print("Word2Vec 模型已保存:")
    print(MODEL_PATH)

    return model



def document_vector(text, model):

    tokens = tokenize(text)

    vectors = []

    for word in tokens:

        if word in model.wv:

            vectors.append(
                model.wv[word]
            )

    # 如果没有任何有效词
    if len(vectors) == 0:

        return np.zeros(
            VECTOR_SIZE,
            dtype=np.float32
        )

    # 对所有有效单词的 Embedding 取平均
    return np.mean(
        vectors,
        axis=0
    )



def build_document_matrix(texts, model, name):

    vectors = []

    for text in tqdm(
        texts,
        desc=name
    ):

        vector = document_vector(
            text,
            model
        )

        vectors.append(vector)

    return np.array(
        vectors,
        dtype=np.float32
    )


def evaluate_model(
    classifier,
    X,
    y,
    dataset_name
):

    predictions = classifier.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    print()
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Macro-F1 : {macro_f1:.6f}"
    )

    return (
        accuracy,
        macro_f1,
        predictions
    )



def main():

    print("=" * 80)
    print("Task 2.3 - Word2Vec Trained on NYT")
    print("=" * 80)

    print()
    print(
        "实验流程：NYT Training Set "
        "-> 训练 100d Word2Vec "
        "-> NYT 文档平均词向量 "
        "-> Logistic Regression"
    )


    train_df, val_df, test_df = load_nyt_data()

    word2vec_model = train_word2vec(
        train_df["text"].tolist()
    )

 
    print()
    print("=" * 80)
    print("生成 NYT 文档向量")
    print("=" * 80)

    print()
    print(
        "方法：对文档中所有有效 Word2Vec Embedding 取平均"
    )

    X_train = build_document_matrix(
        train_df["text"].tolist(),
        word2vec_model,
        "NYT Training Set"
    )

    X_val = build_document_matrix(
        val_df["text"].tolist(),
        word2vec_model,
        "NYT Validation Set"
    )

    X_test = build_document_matrix(
        test_df["text"].tolist(),
        word2vec_model,
        "NYT Test Set"
    )

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    print()
    print("文档向量生成完成！")

    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )


    print()
    print("=" * 80)
    print("训练 Logistic Regression")
    print("=" * 80)

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=SEED
    )

    classifier.fit(
        X_train,
        y_train
    )

    print()
    print(
        "Logistic Regression 训练完成！"
    )


    (
        val_accuracy,
        val_macro_f1,
        _
    ) = evaluate_model(
        classifier,
        X_val,
        y_val,
        "Validation Set"
    )


    (
        test_accuracy,
        test_macro_f1,
        test_predictions
    ) = evaluate_model(
        classifier,
        X_test,
        y_test,
        "NYT Test Set"
    )


    print()
    print(
        "NYT Test Set Classification Report:"
    )

    print()

    print(
        classification_report(
            y_test,
            test_predictions,
            digits=4
        )
    )


    print()
    print("=" * 80)
    print(
        "Word2Vec Trained on NYT 最终结果"
    )
    print("=" * 80)

    print(
        f"Vector Dimension     : {VECTOR_SIZE}"
    )

    print(
        f"Validation Accuracy  : {val_accuracy:.6f}"
    )

    print(
        f"Validation Macro-F1  : {val_macro_f1:.6f}"
    )

    print(
        f"NYT Test Accuracy    : {test_accuracy:.6f}"
    )

    print(
        f"NYT Test Macro-F1    : {test_macro_f1:.6f}"
    )


    results = {

        "method":
            "Word2Vec Trained on NYT",

        "word2vec_training_data":
            "NYT Training Set",

        "vector_size":
            VECTOR_SIZE,

        "window":
            WINDOW,

        "min_count":
            MIN_COUNT,

        "word2vec_epochs":
            EPOCHS,

        "classifier":
            "Logistic Regression",

        "validation_accuracy":
            float(val_accuracy),

        "validation_macro_f1":
            float(val_macro_f1),

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
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("结果已保存:")
    print(RESULT_PATH)

    print()
    print(
        "05_word2vec_nyt.py 执行完成！"
    )



if __name__ == "__main__":
    main()