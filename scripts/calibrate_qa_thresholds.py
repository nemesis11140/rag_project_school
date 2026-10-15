"""高频问答阈值标定。

这里只标定 BM25 候选召回门槛，不代表问题语义等价。
真实语义校验请运行 scripts/evaluate_faq_semantics.py --live。

验收标准：CSV 里每条问题拿自己当 query，必须 100% 命中自己。
在这个前提下，再尽可能放宽阈值来提高改写问法的召回。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.mysql_qa import BM25Retriever, MySQLManager


def main():
    mysql = MySQLManager()
    pairs = mysql.get_all_qa_pairs()
    if not pairs:
        print("MySQL 中没有数据")
        return 1

    questions = [p["question"] for p in pairs]
    ids = [p["id"] for p in pairs]
    retriever = BM25Retriever()
    retriever.load_questions(questions, ids=ids)

    print("=" * 78)
    print("测试 1：自匹配（CSV 原问题查自己）—— 必须 100% 命中")
    print("=" * 78)

    scores, overlaps, self_ok = [], [], 0
    for question in questions:
        candidates = retriever.search_topk(question, top_k=1)
        if not candidates:
            scores.append(0.0)
            overlaps.append(0.0)
            continue
        top = candidates[0]
        scores.append(top["score"])
        overlaps.append(top["keyword_overlap"])
        if top["question"] == question:
            self_ok += 1

    scores = np.array(scores)
    overlaps = np.array(overlaps)

    print(f"  自匹配 top1 正确: {self_ok}/{len(questions)} ({self_ok / len(questions) * 100:.1f}%)")
    print(f"  自匹配 score:   min={scores.min():.3f} p5={np.percentile(scores, 5):.3f} "
          f"中位={np.median(scores):.3f} max={scores.max():.3f}")
    print(f"  自匹配 overlap: min={overlaps.min():.3f} 中位={np.median(overlaps):.3f}")

    print()
    print("=" * 78)
    print("测试 2：不同阈值下的自匹配通过率 / 通过但答错的比例")
    print("=" * 78)
    print(f"  {'score>=':>9} {'overlap>=':>10} {'通过数':>7} {'误答(非自己)':>13} {'漏答':>7}")
    for score_th in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        for ov_th in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            passed_wrong = 0
            missed = 0
            for question, sc, ov in zip(questions, scores, overlaps):
                if sc >= score_th and ov >= ov_th:
                    top = retriever.search_topk(question, top_k=1)[0]
                    if top["question"] != question:
                        passed_wrong += 1
                else:
                    missed += 1
            accepted = len(questions) - missed
            if passed_wrong == 0 and accepted > 0:
                print(f"  {score_th:>9.1f} {ov_th:>10.1f} {accepted:>7} {passed_wrong:>13} {missed:>7}")

    print()
    print("=" * 78)
    print("测试 3：改写问法（同义/口语化）的召回情况")
    print("=" * 78)
    paraphrases = [
        "如何申请奖学金",
        "奖学金咋申请",
        "宿舍几点熄灯",
        "学费多少钱",
        "我弄丢了学生证怎么办",
        "转专业需要什么条件",
        "图书馆借书期限",
        "一卡通怎么补办",
    ]
    for query in paraphrases:
        candidates = retriever.search_topk(query, top_k=1)
        if not candidates:
            print(f"  {query:<22} -> 无候选")
            continue
        top = candidates[0]
        print(f"  {query:<22} -> score={top['score']:6.3f} ov={top['keyword_overlap']:.2f} "
              f"| {top['question']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
