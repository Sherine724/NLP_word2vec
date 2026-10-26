# -*- coding: utf-8 -*-
"""实验二总入口：任务1 分词/词表/训练对 → 任务2 从零训练 → 任务3 语义比较 → 任务4 PCA 画图。

在 my_lab/ 目录下运行：
    python main.py
一条命令跑完全部任务，图片保存在 outputs/。
"""
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))  # 让 main.py 能 import src/ 下的模块

from corpus import (
    build_vocab,
    create_skipgram_pairs,
    load_sentences,
    load_user_dict,
)
from similarity import cosine_similarity, get_vector, load_stopwords, most_similar
from visualize import GROUPS, plot_words
from word2vec import Word2Vec, save_loss_curve

# 超参数集中在这里，改一处即可
EMBED_DIM = 50
WINDOW_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5
TOP_K = 5
SEED = 42


def main():
    np.random.seed(SEED)
    outputs = os.path.join(BASE, "outputs")
    os.makedirs(outputs, exist_ok=True)

    # ---------- 任务 1：分词、词表、训练对 ----------
    print("=" * 55)
    print("任务 1  分词、词表、Skip-gram 训练对")
    print("=" * 55)
    load_user_dict()                      # 必须在 cut 之前
    sentences = load_sentences()
    word_to_id, id_to_word, counter = build_vocab(sentences)
    sentence_ids = [[word_to_id[w] for w in s] for s in sentences]
    pairs = create_skipgram_pairs(sentence_ids, window_size=WINDOW_SIZE)
    V = len(word_to_id)
    print(f"语料来源: corpora/08_童话奇幻（43 句）")
    print(f"句子数 = {len(sentences)}   词表大小 V = {V}   训练对 = {len(pairs)}")
    print("第一句分词:", " / ".join(sentences[0]))
    for w in ["儿童文学", "哥本哈根", "捏黏土"]:
        print(f"  主题词「{w}」在词表中: {'通过' if w in word_to_id else '未通过'}")

    # ---------- 任务 2：从零训练 Skip-gram + 负采样 ----------
    print("\n" + "=" * 55)
    print("任务 2  从零训练 Skip-gram + 负采样")
    print("=" * 55)
    model = Word2Vec(
        vocab_size=V,
        embedding_dim=EMBED_DIM,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        counter=counter,
    )
    t0 = time.time()
    losses = model.train(
        pairs, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        negative_samples=NEGATIVE_SAMPLES,
    )
    print(f"训练用时 {time.time() - t0:.1f} 秒")
    print(f"第 1 轮 loss = {losses[0]:.4f}（瞎猜基线 ln2 = 0.6931）")
    print(f"最后一轮 loss = {losses[-1]:.4f}")
    print(f"loss 是否明显下降: {'通过' if losses[-1] < losses[0] - 0.05 else '未通过'}")
    save_loss_curve(losses, os.path.join(outputs, "loss_curve.png"))
    W = model.W   # 之后只用输入侧词向量 W

    # ---------- 任务 3：余弦相似度与语义比较 ----------
    print("\n" + "=" * 55)
    print("任务 3  余弦相似度与语义比较")
    print("=" * 55)
    stopwords = load_stopwords()

    def sim(w1, w2):
        return cosine_similarity(get_vector(w1, word_to_id, W),
                                 get_vector(w2, word_to_id, W))

    comparisons = [
        ("王子", "公主", "王子", "企鹅"),
        ("哥本哈根", "布拉格", "哥本哈根", "热可可"),
    ]
    for ra, rb, ua, ub in comparisons:
        s_rel, s_unrel = sim(ra, rb), sim(ua, ub)
        ok = s_rel > s_unrel + 0.05
        print(f"sim({ra},{rb}) = {s_rel:.4f}   sim({ua},{ub}) = {s_unrel:.4f}"
              f"   相关>无关+0.05: {'通过' if ok else '未通过'}")

    for q in ["王子", "哥本哈根"]:
        hits = most_similar(q, word_to_id, W, stopwords, top_k=TOP_K)
        print(f"\n与「{q}」最相似的 {TOP_K} 个词:")
        for w, s in hits:
            print(f"  {w:8s}  {s:.4f}")

    # ---------- 任务 4：PCA 可视化 ----------
    print("\n" + "=" * 55)
    print("任务 4  PCA 可视化")
    print("=" * 55)
    plot_words(GROUPS, word_to_id, W, os.path.join(outputs, "pca_words.png"))

    print("\n全部完成。outputs/ 下应有: loss_curve.png、pca_words.png")


if __name__ == "__main__":
    main()
