import os
import json
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report




BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "nyt_train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "nyt_val.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "nyt_test.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)




print("=" * 70)
print("Task 1 - Bag of Words + Logistic Regression")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\n数据读取完成：")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

X_train_text = train_df["text"].astype(str)
y_train = train_df["label"].astype(str)

X_val_text = val_df["text"].astype(str)
y_val = val_df["label"].astype(str)

X_test_text = test_df["text"].astype(str)
y_test = test_df["label"].astype(str)




def evaluate_model(model_name, model, X_val, X_test):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    # Validation
    val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro"
    )

    print("\nValidation Set")
    print(f"Accuracy : {val_accuracy:.6f}")
    print(f"Macro-F1 : {val_macro_f1:.6f}")

    # Test
    test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro"
    )

    print("\nTest Set")
    print(f"Accuracy : {test_accuracy:.6f}")
    print(f"Macro-F1 : {test_macro_f1:.6f}")

    print("\nTest Set Classification Report:")
    print(
        classification_report(
            y_test,
            test_pred,
            digits=4
        )
    )

    return {
        "method": model_name,
        "validation_accuracy": float(val_accuracy),
        "validation_macro_f1": float(val_macro_f1),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1)
    }




print("\n" + "#" * 70)
print("Experiment 1: Binary Bag of Words")
print("#" * 70)

binary_vectorizer = CountVectorizer(
    binary=True
)

print("\n正在建立 Binary BoW 词表...")

X_train_binary = binary_vectorizer.fit_transform(X_train_text)
X_val_binary = binary_vectorizer.transform(X_val_text)
X_test_binary = binary_vectorizer.transform(X_test_text)

print("词表大小：", len(binary_vectorizer.vocabulary_))
print("Train 特征矩阵：", X_train_binary.shape)
print("Validation 特征矩阵：", X_val_binary.shape)
print("Test 特征矩阵：", X_test_binary.shape)

print("\n正在训练 Logistic Regression...")

binary_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

binary_model.fit(
    X_train_binary,
    y_train
)

binary_result = evaluate_model(
    "Binary Bag of Words",
    binary_model,
    X_val_binary,
    X_test_binary
)




print("\n" + "#" * 70)
print("Experiment 2: Word Frequency")
print("#" * 70)

frequency_vectorizer = CountVectorizer(
    binary=False
)

print("\n正在建立 Word Frequency 词表...")

X_train_frequency = frequency_vectorizer.fit_transform(X_train_text)
X_val_frequency = frequency_vectorizer.transform(X_val_text)
X_test_frequency = frequency_vectorizer.transform(X_test_text)

print("词表大小：", len(frequency_vectorizer.vocabulary_))
print("Train 特征矩阵：", X_train_frequency.shape)
print("Validation 特征矩阵：", X_val_frequency.shape)
print("Test 特征矩阵：", X_test_frequency.shape)

print("\n正在训练 Logistic Regression...")

frequency_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

frequency_model.fit(
    X_train_frequency,
    y_train
)

frequency_result = evaluate_model(
    "Word Frequency",
    frequency_model,
    X_val_frequency,
    X_test_frequency
)




binary_result_path = os.path.join(
    RESULTS_DIR,
    "bow_binary.json"
)

frequency_result_path = os.path.join(
    RESULTS_DIR,
    "bow_frequency.json"
)

with open(binary_result_path, "w", encoding="utf-8") as f:
    json.dump(
        binary_result,
        f,
        indent=4,
        ensure_ascii=False
    )

with open(frequency_result_path, "w", encoding="utf-8") as f:
    json.dump(
        frequency_result,
        f,
        indent=4,
        ensure_ascii=False
    )



print("\n" + "=" * 70)
print("Task 1 最终结果")
print("=" * 70)

print(
    f"{'Method':<25}"
    f"{'Test Accuracy':>18}"
    f"{'Test Macro-F1':>18}"
)

print("-" * 61)

print(
    f"{'Binary Bag of Words':<25}"
    f"{binary_result['test_accuracy']:>18.6f}"
    f"{binary_result['test_macro_f1']:>18.6f}"
)

print(
    f"{'Word Frequency':<25}"
    f"{frequency_result['test_accuracy']:>18.6f}"
    f"{frequency_result['test_macro_f1']:>18.6f}"
)

print("\n结果已保存：")
print(binary_result_path)
print(frequency_result_path)

print("\n02_bow.py 执行完成！")