# -*- coding: utf-8 -*-
"""任务 4：用 PCA 把词向量降到二维，画带中文标签的分组散点图。

对照阅读：zhword2vec/src/visualize.py（pca_2d、plot_words）
在 my_lab/src/ 下运行：
    python visualize.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 按语料的四个语义主题分组，画图时用不同颜色/标记
GROUPS = {
    "王室": ["王子", "公主", "男孩", "女孩", "王国", "守卫"],
    "城市": ["哥本哈根", "布拉格", "丹麦", "捷克", "梦都"],
    "爱好": ["企鹅", "北极熊", "搭积木", "捏黏土", "放风筝", "热可可"],
    "艺术": ["绘本", "童话", "素描", "画漫画", "儿童文学"],
}
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
MARKERS = ["o", "s", "^", "D"]


def setup_chinese_font():
    """Windows 上不设中文字体，标签会变成方框。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def pca_2d(X):
    """把矩阵 X（每行一个词向量）降到 2 维。

    三步：去均值 → 求协方差矩阵的特征分解 → 投影到前 2 个特征向量。
    特征向量按特征值从大到小排：特征值 = 该方向上的方差，
    取最大的两个方向，就是「保留信息最多」的二维平面。
    """
    X = X - X.mean(axis=0)                       # 中心化：均值挪到原点
    cov = np.cov(X, rowvar=False)                # 50×50 协方差矩阵
    vals, vecs = np.linalg.eigh(cov)             # eigh 专用于对称矩阵，特征值升序
    order = np.argsort(vals)[::-1]               # 翻转成降序
    return X @ vecs[:, order[:2]]                # 投影：得到 n×2 坐标


def plot_words(groups, word_to_id, W, save_path, show=False):
    """对 groups 里的词做 PCA 并按组画散点图，保存 PNG。

    注意：PCA 只用「被画的这些词」估计主成分方向——
    如果用全部 49 个词，方差最大的方向会被高频枢纽词（我/喜欢/很）主导，
    语义组会挤成一团。
    """
    setup_chinese_font()
    words = [w for ws in groups.values() for w in ws if w in word_to_id]
    X = np.array([W[word_to_id[w]] for w in words])
    coords = pca_2d(X)

    plt.figure(figsize=(9, 7))
    for (name, gwords), color, marker in zip(groups.items(), COLORS, MARKERS):
        idx = [i for i, w in enumerate(words) if w in set(gwords)]
        plt.scatter(coords[idx, 0], coords[idx, 1],
                    c=color, marker=marker, s=110, label=name, zorder=3)
        for i in idx:
            plt.annotate(words[i], (coords[i, 0], coords[i, 1]),
                         textcoords="offset points", xytext=(8, 6), fontsize=11)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("词向量 PCA（二维距离仅供参考，硬指标是高维余弦）")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130)
    if show:
        plt.show()
    plt.close()
    print(f"散点图已保存: {save_path}（共画了 {len(words)} 个词）")


if __name__ == "__main__":
    from corpus import (
        build_vocab,
        create_skipgram_pairs,
        load_sentences,
        load_user_dict,
    )
    from word2vec import Word2Vec

    # 同种子同参数 => W 与任务 2/3 完全一致
    np.random.seed(42)
    load_user_dict()
    sentences = load_sentences()
    word_to_id, id_to_word, counter = build_vocab(sentences)
    sentence_ids = [[word_to_id[w] for w in s] for s in sentences]
    pairs = create_skipgram_pairs(sentence_ids, window_size=2)

    print("重新训练模型（同种子同参数）...")
    model = Word2Vec(len(word_to_id), 50, word_to_id, id_to_word, counter)
    model.train(pairs, epochs=100, learning_rate=0.01, negative_samples=5)

    out = os.path.join(BASE_DIR, "outputs", "pca_words.png")
    plot_words(GROUPS, word_to_id, model.W, out, show=False)
