# -*- coding: utf-8 -*-
"""任务 2：从零实现 Skip-gram + 负采样（纯 NumPy）。

对照阅读：zhword2vec/src/word2vec.py
在 my_lab/src/ 下运行：
    python word2vec.py
"""
import math
import os

import numpy as np


def sigmoid(x):
    """把任意实数压到 (0,1)，当作「target 是真上下文的概率」。

    clip(-10,10) 防止 exp 溢出：exp(1000) 会变成 inf。
    截断几乎不影响数值，因为 sigmoid(10)≈0.99995 已经贴着 1 了。
    """
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, word_to_id, id_to_word, counter):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        # W：输入侧（中心词）向量表，训练完查相似词、画图都用它
        # U：输出侧（上下文）参数表
        # W 用小随机数打破对称；U 全零 => 初始点积=0 => sigmoid=0.5
        # => 每个二分类样本 loss = ln2 ≈ 0.693，这就是「瞎猜基线」
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.U = np.zeros((vocab_size, embedding_dim))
        # 负采样噪声分布：P(w) ∝ freq(w)^0.75，再归一化成概率
        freqs = np.array(
            [counter[id_to_word[i]] for i in range(vocab_size)], dtype=float
        )
        freqs = freqs ** 0.75
        self.noise_dist = freqs / freqs.sum()

    def sample_negative_words(self, positive_id, k):
        """按噪声分布抽 k 个负样本，避开当前正样本。"""
        negatives = []
        while len(negatives) < k:
            idx = np.random.choice(len(self.noise_dist), p=self.noise_dist)
            if idx != positive_id:
                negatives.append(idx)
        return negatives

    def train_pair(self, center_id, target_id, label, learning_rate):
        """训练一个二分类样本：label=1（真上下文）/ label=0（负样本）。

        模型预测 prob = sigmoid(U[target]·W[center])，
        目标是：正样本 prob→1，负样本 prob→0。
        error = prob - label 恰好是二元交叉熵对点积的梯度：
          正样本 prob<1 时 error<0，更新让点积变大；
          负样本 prob>0 时 error>0，更新让点积变小。
        先把两个向量 copy 下来：U 更新后，W 的梯度还要用「更新前」的 U，
        否则第二个更新用的是被污染的值，梯度就错了。
        """
        center_vec = self.W[center_id].copy()
        target_vec = self.U[target_id].copy()
        prob = sigmoid(np.dot(target_vec, center_vec))
        error = prob - label
        self.U[target_id] -= learning_rate * error * center_vec
        self.W[center_id] -= learning_rate * error * target_vec
        # 本样本的二元交叉熵损失（+1e-10 防 log(0)）
        return -(label * np.log(prob + 1e-10)
                 + (1 - label) * np.log(1 - prob + 1e-10))

    def train(self, pairs, epochs, learning_rate, negative_samples):
        """逐对训练：每个正样本配 K 个负样本。返回每轮平均 loss 列表。

        每轮先 shuffle，避免样本顺序带来的更新偏差。
        平均 loss 的分母是 len(pairs)*(1+K)：每个训练对实际做了 1+K 次二分类。
        """
        epoch_losses = []
        for epoch in range(1, epochs + 1):
            np.random.shuffle(pairs)
            total_loss = 0.0
            for center_id, context_id in pairs:
                total_loss += self.train_pair(
                    center_id, context_id, 1, learning_rate
                )
                for neg_id in self.sample_negative_words(context_id, negative_samples):
                    total_loss += self.train_pair(
                        center_id, neg_id, 0, learning_rate
                    )
            avg = total_loss / (len(pairs) * (1 + negative_samples))
            epoch_losses.append(avg)
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch {epoch:3d}  loss = {avg:.4f}")
        return epoch_losses


def save_loss_curve(losses, path):
    """把每轮平均 loss 画成曲线存图（报告用）。"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Skip-gram + Negative Sampling Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=130)
    plt.close()
    print("损失曲线已保存:", path)


if __name__ == "__main__":
    import time

    from corpus import (
        BASE_DIR,
        build_vocab,
        create_skipgram_pairs,
        load_sentences,
        load_user_dict,
    )

    np.random.seed(42)  # 固定种子，结果可复现

    load_user_dict()
    sentences = load_sentences()
    word_to_id, id_to_word, counter = build_vocab(sentences)
    sentence_ids = [[word_to_id[w] for w in s] for s in sentences]
    pairs = create_skipgram_pairs(sentence_ids, window_size=2)
    V = len(word_to_id)
    print(f"V = {V}，训练对 = {len(pairs)}")

    model = Word2Vec(
        vocab_size=V,
        embedding_dim=50,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        counter=counter,
    )

    print("\n=== 任务 2 验收：训练（观察 loss 是否逐轮下降）===")
    t0 = time.time()
    losses = model.train(pairs, epochs=100, learning_rate=0.01, negative_samples=5)
    print(f"训练用时 {time.time() - t0:.1f} 秒")

    print("\n--- 验收指标 ---")
    print(f"第 1 轮 loss = {losses[0]:.4f}  （瞎猜基线 ln2 = {math.log(2):.4f}，应接近）")
    print(f"最后一轮 loss = {losses[-1]:.4f}")
    print(f"loss 是否明显下降: {'通过' if losses[-1] < losses[0] - 0.05 else '未通过'}")

    save_loss_curve(losses, os.path.join(BASE_DIR, "outputs", "loss_curve.png"))
