# 新闻文本分类实验

## 一、项目简介

本项目为自然语言处理课程大作业，使用 NYT 新闻数据集完成文本分类任务，对比六种文本分类方法的效果，并使用 Accuracy 和 Macro-F1 进行评价。

## 二、实验方法

本实验共实现六种分类方法：

1. Binary Bag of Words（二值词袋模型）
2. Word Frequency（词频模型）
3. GloVe 100d（预训练词向量）
4. Word2Vec - AG News
5. Word2Vec - NYT
6. BERT-base-uncased

## 三、项目文件

- `code/`：实验 Python 代码
- `data/`：实验数据集
- `png/`：实验过程截图
- `results/`：实验结果及对比图
- `glove/`：GloVe 词向量文件（需自行下载）
- `models/`：训练模型（未上传）

## 四、运行环境

使用 Python 和 Conda 环境 `textcls`。

主要依赖：PyTorch、Transformers、Gensim、Scikit-learn、Pandas、NumPy、Matplotlib。

安装依赖：

```bash
pip install torch transformers gensim scikit-learn pandas numpy matplotlib tqdm
```

## 五、运行方法

首先激活环境：

```bash
conda activate textcls
```

在项目根目录依次运行：

```bash
python code/01_prepare_data.py
python code/02_bow.py
python code/03_glove.py
python code/04_word2vec_ag.py
python code/05_word2vec_nyt.py
python code/06_bert.py
python code/07_compare_results.py
```

GloVe 词向量下载地址：

https://nlp.stanford.edu/data/glove.6B.zip

使用其中的 `glove.6B.100d.txt` 文件。

## 六、实验结果

| 分类方法 | Accuracy | Macro-F1 |
|---|---|---|
| Binary Bag of Words | 0.990451 | 0.976383 |
| Word Frequency | 0.989583 | 0.974178 |
| GloVe 100d | 0.985243 | 0.964617 |
| Word2Vec - AG News | 0.980035 | 0.950803 |
| Word2Vec - NYT | 0.986979 | 0.967214 |
| BERT-base-uncased | 0.979167 | 0.951085 |

以上结果均来自 NYT Test Set。

## 七、实验总结

六种方法均取得了较好的分类效果，其中 Binary Bag of Words 表现最好，Accuracy 达到 99.05%，Macro-F1 达到 97.64%。

具体实验过程、结果分析及截图见项目中的实验报告 PDF 文件。