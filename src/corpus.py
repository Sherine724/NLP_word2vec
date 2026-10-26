# -*- coding: utf-8 -*-
"""任务 1：读语料、jieba 分词、建词表、构造 Skip-gram 训练对。

语料来源：nlp-course chapter03/zhword2vec/corpora/08_童话奇幻
在 my_lab/src/ 下运行：
    python corpus.py
"""
import logging
import os
from collections import Counter

import jieba

jieba.setLogLevel(logging.WARNING)  # 不打印 jieba 自己的加载日志

# corpus.py 在 src/ 里，上一级才是项目根目录 my_lab/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_DICT_PATH = os.path.join(BASE_DIR, "data", "user_dict.txt")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus.txt")

# 中英文标点都当分隔符处理，不进词表
PUNCT = set("，。！？、；：,.!?;:（）()“”\"'…—")


def load_user_dict(path=USER_DICT_PATH):
    """加载主题词典，必须在 jieba.cut 之前调用。

    为什么：jieba 默认词典里没有「哥本哈根」「儿童文学」这类词，
    会被切成「哥本/哈根」，主题词就进不了词表。
    """
    jieba.load_userdict(path)


def tokenize_line(line):
    """把一句话切成词列表，去掉空白和标点。"""
    return [w for w in jieba.cut(line.strip()) if w.strip() and w not in PUNCT]


def load_sentences(corpus_path=CORPUS_PATH):
    """按行读语料（一行一句），返回 list[list[str]]。

    为什么保留句子边界：Skip-gram 窗口不能跨句，
    否则上一句末尾的词和下一句开头的词会被当成邻居，造出不存在的共现。
    """
    sentences = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences.append(tokenize_line(line))
    return sentences


def build_vocab(sentences):
    """统计词频，建立 词<->编号 双向映射。

    排序规则 (-词频, 汉字Unicode)：高频在前、同频按字典序，
    保证每次运行词表完全一样（可复现）。
    """
    counter = Counter()
    for words in sentences:
        counter.update(words)
    ordered = sorted(counter, key=lambda w: (-counter[w], w))
    word_to_id = {w: i for i, w in enumerate(ordered)}
    id_to_word = {i: w for i, w in enumerate(ordered)}
    return word_to_id, id_to_word, counter


def create_skipgram_pairs(sentence_ids, window_size=2):
    """对每句话生成 (中心词编号, 上下文词编号)。

    双层循环等价于「窗口在句内滑动」：
    i 是中心词位置，j 扫它左右各 window_size 个位置，
    max/min 截断到句子边界，j == i 跳过自己。
    """
    pairs = []
    for ids in sentence_ids:
        n = len(ids)
        for i in range(n):
            for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
                if j != i:
                    pairs.append((ids[i], ids[j]))
    return pairs


if __name__ == "__main__":
    load_user_dict()  # 一定要在 cut 之前
    sentences = load_sentences()
    word_to_id, id_to_word, counter = build_vocab(sentences)
    V = len(word_to_id)

    print("=== 任务 1 验收 ===")
    print(f"句子数 = {len(sentences)}")
    print(f"词表大小 V = {V}")
    print("第一句分词：", " / ".join(sentences[0]))

    # 检查 1：主题词必须是完整的一个词
    print("\n--- 主题词完整性 ---")
    for w in ["儿童文学", "哥本哈根", "捏黏土", "叙事艺术"]:
        ok = w in word_to_id
        print(f"  「{w}」在词表中: {'通过' if ok else '未通过'}  (id={word_to_id.get(w)})")

    # 检查 2：手算训练对——3 个词的句子，窗口 2 应得 6 对
    print("\n--- 手算验收：我 / 喜欢 / 绘本 ---")
    toy = ["我", "喜欢", "绘本"]
    toy_pairs = create_skipgram_pairs([[word_to_id[w] for w in toy]], window_size=2)
    print(f"训练对数 = {len(toy_pairs)}（应为 6）")
    for c, x in toy_pairs:
        print(f"  {id_to_word[c]} → {id_to_word[x]}")

    # 全量训练对
    sentence_ids = [[word_to_id[w] for w in s] for s in sentences]
    pairs = create_skipgram_pairs(sentence_ids, window_size=2)
    print(f"\n总训练对数量 = {len(pairs)}")
    print("前 5 对（中心词 → 上下文词）：")
    for c, x in pairs[:5]:
        print(f"  {id_to_word[c]} → {id_to_word[x]}")