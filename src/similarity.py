# -*- coding: utf-8 -*-
"""任务 3：手写余弦相似度 + 相似词查询 + 语义比较。

对照阅读：zhword2vec/src/similarity.py（get_vector、cosine_similarity）；
近邻查询的写法参考 zhword2vec/src/word2vec.py 里的 most_similar。
在 my_lab/src/ 下运行：
    python similarity.py
"""
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOPWORDS_PATH = os.path.join(BASE_DIR, "data", "stopwords.txt")


def get_vector(word, word_to_id, W):
    """查表：词 → id → W 的第 id 行，就是这个词的向量。

    用输入侧 W 而不是输出侧 U（思考题 Q1）。
    """
    return W[word_to_id[word]]


def cosine_similarity(a, b):
    """余弦相似度 = 点积 / (|a| × |b|)，即两向量夹角的余弦。

    手写实现，不调用现成函数。除以两个模长是为了「只看方向不看长度」。
    分母为 0（零向量）时返回 0.0，避免除零崩溃。
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def load_stopwords(path=STOPWORDS_PATH):
    """停用词（的/是/很…）跟所有词都共现，会霸榜相似度结果，查询时过滤。

    注意：只在「查询」时过滤，训练时不过滤——停用词也是有效的上下文信号。
    """
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def most_similar(word, word_to_id, W, stopwords=None, top_k=5):
    """返回 [(词, 相似度), ...]，排除自身和停用词，按相似度降序。"""
    if word not in word_to_id:
        raise KeyError(f"词表中没有「{word}」")
    query = get_vector(word, word_to_id, W)
    scores = []
    for w, idx in word_to_id.items():
        if w == word:
            continue
        if stopwords and w in stopwords:
            continue
        scores.append((w, cosine_similarity(query, W[idx])))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    import time

    from corpus import (
        build_vocab,
        create_skipgram_pairs,
        load_sentences,
        load_user_dict,
    )
    from word2vec import Word2Vec

    # 与任务 2 完全相同的种子、语料、超参数 => 训出的 W 一模一样，
    # 所以本文件的输出数字和任务 2 那次训练完全一致、可复现。
    np.random.seed(42)
    load_user_dict()
    sentences = load_sentences()
    word_to_id, id_to_word, counter = build_vocab(sentences)
    sentence_ids = [[word_to_id[w] for w in s] for s in sentences]
    pairs = create_skipgram_pairs(sentence_ids, window_size=2)

    print("重新训练模型（同种子同参数，结果与任务 2 一致）...")
    model = Word2Vec(len(word_to_id), 50, word_to_id, id_to_word, counter)
    t0 = time.time()
    model.train(pairs, epochs=100, learning_rate=0.01, negative_samples=5)
    print(f"训练用时 {time.time() - t0:.1f} 秒")
    W = model.W

    def sim(w1, w2):
        return cosine_similarity(get_vector(w1, word_to_id, W),
                                 get_vector(w2, word_to_id, W))

    print("\n=== 任务 3 验收 ===")
    print("\n--- 检查 1：余弦的数学性质 ---")
    v = np.array([3.0, 4.0])
    print(f"  cos(v, v)   = {cosine_similarity(v, v):.4f}   （应为 1）")
    print(f"  cos(v, -v)  = {cosine_similarity(v, -v):.4f}   （应为 -1）")
    print(f"  cos(v, 正交) = {cosine_similarity(v, np.array([-4.0, 3.0])):.4f}   （应为 0）")

    print("\n--- 检查 2：语义比较（相关对应明显高于无关对）---")
    comparisons = [
        ("王子", "公主", "王子", "企鹅"),
        ("哥本哈根", "布拉格", "哥本哈根", "热可可"),
    ]
    for ra, rb, ua, ub in comparisons:
        s_rel, s_unrel = sim(ra, rb), sim(ua, ub)
        ok = s_rel > s_unrel + 0.05
        print(f"  sim({ra},{rb}) = {s_rel:.4f}   sim({ua},{ub}) = {s_unrel:.4f}"
              f"   相关>无关+0.05: {'通过' if ok else '未通过'}")
    print("\n  更多相关对（报告素材）：")
    for a, b in [("企鹅", "北极熊"), ("绘本", "童话")]:
        print(f"  sim({a},{b}) = {sim(a, b):.4f}")

    print("\n--- 检查 3：相似词查询（停用词已过滤）---")
    stopwords = load_stopwords()
    for q in ["王子", "哥本哈根", "企鹅"]:
        hits = most_similar(q, word_to_id, W, stopwords, top_k=5)
        print(f"\n  与「{q}」最相似：")
        for w, s in hits:
            print(f"    {w:8s}  {s:.4f}")
