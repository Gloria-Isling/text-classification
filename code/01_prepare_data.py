import os
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NYT_PATH = os.path.join(BASE_DIR, "data", "nyt.csv")

TRAIN_PATH = os.path.join(BASE_DIR, "data", "nyt_train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "nyt_val.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "nyt_test.csv")


RANDOM_STATE = 42


print("=" * 60)
print("读取 NYT 数据集")
print("=" * 60)

df = pd.read_csv(NYT_PATH)

print("数据形状：", df.shape)
print("列名：", list(df.columns))

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("nyt.csv 必须包含 text 和 label 两列")



print("\n开始清理数据...")

# 删除 text 或 label 为空的数据
df = df.dropna(subset=["text", "label"]).copy()

# 转换为字符串
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str)

# 删除纯空白文本
df = df[df["text"].str.strip() != ""].copy()

# 重置索引
df = df.reset_index(drop=True)

print("清理后数据量：", len(df))



print("\n完整数据集类别分布：")
print(df["label"].value_counts())

print("\n完整数据集类别比例：")
print(df["label"].value_counts(normalize=True))



train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)



val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)


# 重置索引
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)



print("\n" + "=" * 60)
print("数据集划分完成")
print("=" * 60)

print(f"总数据量：       {len(df)}")
print(f"Training Set：  {len(train_df)} ({len(train_df)/len(df):.2%})")
print(f"Validation Set：{len(val_df)} ({len(val_df)/len(df):.2%})")
print(f"Test Set：      {len(test_df)} ({len(test_df)/len(df):.2%})")


print("\nTraining Set 类别分布：")
print(train_df["label"].value_counts())

print("\nValidation Set 类别分布：")
print(val_df["label"].value_counts())

print("\nTest Set 类别分布：")
print(test_df["label"].value_counts())



train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
val_df.to_csv(VAL_PATH, index=False, encoding="utf-8")
test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")


print("\n" + "=" * 60)
print("文件保存成功")
print("=" * 60)

print(TRAIN_PATH)
print(VAL_PATH)
print(TEST_PATH)

print("\n01_prepare_data.py 执行完成！")