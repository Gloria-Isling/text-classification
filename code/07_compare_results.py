import os
import json
import pandas as pd
import matplotlib.pyplot as plt



BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULT_DIR = os.path.join(BASE_DIR, "results")

OUTPUT_CSV = os.path.join(
    RESULT_DIR,
    "all_results.csv"
)

ACCURACY_FIG = os.path.join(
    RESULT_DIR,
    "accuracy_comparison.png"
)

F1_FIG = os.path.join(
    RESULT_DIR,
    "macro_f1_comparison.png"
)



RESULT_FILES = [
    (
        "Binary Bag of Words",
        "bow_binary.json"
    ),
    (
        "Word Frequency",
        "bow_frequency.json"
    ),
    (
        "GloVe 100d",
        "glove.json"
    ),
    (
        "Word2Vec - AG News",
        "word2vec_ag.json"
    ),
    (
        "Word2Vec - NYT",
        "word2vec_nyt.json"
    ),
    (
        "BERT-base-uncased",
        "bert.json"
    ),
]



print("=" * 80)
print("Final Experiment Results Comparison")
print("=" * 80)

print("\n读取实验结果...\n")



all_results = []

for method_name, filename in RESULT_FILES:

    file_path = os.path.join(
        RESULT_DIR,
        filename
    )

    print(f"读取: {filename}")

    if not os.path.exists(file_path):

        print(
            f"错误：找不到文件：{file_path}"
        )

        raise FileNotFoundError(
            f"缺少实验结果文件：{filename}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    
    if "test_accuracy" in data:
        accuracy = data["test_accuracy"]

    elif "accuracy" in data:
        accuracy = data["accuracy"]

    else:
        raise KeyError(
            f"{filename} 中找不到 Test Accuracy"
        )


    if "test_macro_f1" in data:
        macro_f1 = data["test_macro_f1"]

    elif "macro_f1" in data:
        macro_f1 = data["macro_f1"]

    else:
        raise KeyError(
            f"{filename} 中找不到 Test Macro-F1"
        )


    all_results.append(
        {
            "Method": method_name,
            "Test Accuracy": float(accuracy),
            "Test Macro-F1": float(macro_f1)
        }
    )


print("\n所有实验结果读取成功！")



results_df = pd.DataFrame(
    all_results
)



print("\n" + "=" * 80)
print("所有模型在 NYT Test Set 上的最终结果")
print("=" * 80)

print()

print(
    f"{'Method':<28}"
    f"{'Test Accuracy':>18}"
    f"{'Test Macro-F1':>18}"
)

print("-" * 64)


for _, row in results_df.iterrows():

    print(
        f"{row['Method']:<28}"
        f"{row['Test Accuracy']:>18.6f}"
        f"{row['Test Macro-F1']:>18.6f}"
    )



best_accuracy_index = (
    results_df["Test Accuracy"].idxmax()
)

best_f1_index = (
    results_df["Test Macro-F1"].idxmax()
)


best_accuracy_method = results_df.loc[
    best_accuracy_index,
    "Method"
]

best_accuracy = results_df.loc[
    best_accuracy_index,
    "Test Accuracy"
]


best_f1_method = results_df.loc[
    best_f1_index,
    "Method"
]

best_f1 = results_df.loc[
    best_f1_index,
    "Test Macro-F1"
]


print("\n" + "=" * 80)
print("Best Results")
print("=" * 80)

print(
    f"\nBest Test Accuracy : "
    f"{best_accuracy_method}"
)

print(
    f"Accuracy           : "
    f"{best_accuracy:.6f}"
)

print(
    f"\nBest Test Macro-F1 : "
    f"{best_f1_method}"
)

print(
    f"Macro-F1           : "
    f"{best_f1:.6f}"
)



accuracy_rank = results_df.sort_values(
    by="Test Accuracy",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 80)
print("Ranking by Test Accuracy")
print("=" * 80)

for index, row in accuracy_rank.iterrows():

    print(
        f"{index + 1}. "
        f"{row['Method']:<25} "
        f"{row['Test Accuracy']:.6f}"
    )



f1_rank = results_df.sort_values(
    by="Test Macro-F1",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 80)
print("Ranking by Test Macro-F1")
print("=" * 80)

for index, row in f1_rank.iterrows():

    print(
        f"{index + 1}. "
        f"{row['Method']:<25} "
        f"{row['Test Macro-F1']:.6f}"
    )



results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("保存最终结果表")
print("=" * 80)

print("\n已保存：")
print(OUTPUT_CSV)



plt.figure(
    figsize=(11, 6)
)

bars = plt.bar(
    results_df["Method"],
    results_df["Test Accuracy"]
)

plt.title(
    "NYT Test Set - Accuracy Comparison",
    fontsize=14
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Accuracy"
)

# 为了突出模型之间的差异
plt.ylim(
    0.94,
    1.00
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)


# 在柱状图上显示数值
for bar, value in zip(
    bars,
    results_df["Test Accuracy"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.0005,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )


plt.tight_layout()

plt.savefig(
    ACCURACY_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()



plt.figure(
    figsize=(11, 6)
)

bars = plt.bar(
    results_df["Method"],
    results_df["Test Macro-F1"]
)

plt.title(
    "NYT Test Set - Macro-F1 Comparison",
    fontsize=14
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Macro-F1"
)

plt.ylim(
    0.90,
    1.00
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)


for bar, value in zip(
    bars,
    results_df["Test Macro-F1"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.001,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )


plt.tight_layout()

plt.savefig(
    F1_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()



print("\n" + "=" * 80)
print("对比图生成完成")
print("=" * 80)

print("\nAccuracy 对比图：")
print(ACCURACY_FIG)

print("\nMacro-F1 对比图：")
print(F1_FIG)


print("\n" + "=" * 80)
print("实验最终汇总")
print("=" * 80)

print(
    f"\n共完成 {len(results_df)} 组实验。"
)

print(
    f"\n最高 Accuracy："
    f"{best_accuracy_method} "
    f"({best_accuracy:.6f})"
)

print(
    f"最高 Macro-F1："
    f"{best_f1_method} "
    f"({best_f1:.6f})"
)

print("\n生成文件：")

print(
    "1.",
    OUTPUT_CSV
)

print(
    "2.",
    ACCURACY_FIG
)

print(
    "3.",
    F1_FIG
)

print("\n07_compare_results.py 执行完成！")