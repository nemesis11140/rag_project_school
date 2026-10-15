"""统一的中文分词 / 停用词 / query 归一化工具。

高频繁问答（mysql_qa）与 RAG 检索（rag_qa）共用这里的实现，
避免出现两份停用词表不一致、tokenize 行为不同的情况。
"""
import re
from typing import Iterable, List, Optional

import jieba

# ---------------------------------------------------------------------------
# 停用词
# ---------------------------------------------------------------------------

# 通用中文虚词：对 BM25 几乎无区分度，只会稀释信号
STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "是", "在", "有", "为", "被", "把", "对",
    "就", "都", "也", "还", "又", "而", "则", "等", "个", "些", "之", "其",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "这", "那", "这个", "那个", "这些", "那些", "这里", "那里",
    "一个", "一些", "什么", "怎么", "怎样", "如何", "哪些", "哪个", "哪里",
    "吗", "吧", "呢", "啊", "呀", "哦", "哈", "嗯", "嘛", "么",
    "请", "可以", "能否", "是否", "需要", "如果", "但是", "然后", "因为", "所以",
}

# 疑问词：只在对"问题"做匹配时剔除（让「如何申请奖学金」≈「怎么申请奖学金」）
QUESTION_WORDS = {
    "什么", "怎么", "怎样", "如何", "哪些", "哪个", "哪里", "多少", "多久",
    "何时", "为什么", "干嘛", "干啥", "咋", "咋办", "请问", "吗", "呢", "吧", "？", "?",
}

# 校园领域词典：保证「奖学金」「转专业」这类词不会被 jieba 切碎。
#
# 只收录"一定是完整词"的条目。像「门禁时间」这种由两个常见词拼起来的组合
# 不要加：查询侧说「门禁」而文档侧被切成「门禁时间」时，两侧 token 不对称，
# 覆盖率会凭空掉一档（实测「宿舍门禁是几点」覆盖率从 1.00 掉到 0.67）。
DOMAIN_WORDS = [
    "打架斗殴", "酗酒闹事", "赌博吸毒", "违纪违规", "学术不端",
    "处分警告", "严重警告", "记过开除", "留校察看", "开除学籍",
    "国家奖学金", "励志奖学金", "助学金", "助学贷款", "勤工俭学", "勤工助学",
    "困难补助", "缓考", "免修", "免听", "缓修", "补考", "重修",
    "转专业", "退学", "入学", "报到", "选课", "答辩", "实习",
    "学生证", "校园卡", "一卡通", "借书证", "图书卡",
    "四六级", "考研", "保研", "推免", "留学", "出国",
    "校医院", "心理咨询", "心理健康", "医保",
    "学籍管理", "成绩单", "毕业论文", "毕业设计",
]

_words_loaded = False


def load_domain_words():
    """把校园领域词注册进 jieba 词典（只做一次）。"""
    global _words_loaded
    if _words_loaded:
        return
    for word in DOMAIN_WORDS:
        jieba.add_word(word)
    _words_loaded = True


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

# 同义词 -> 标准词。使用"左到右、不回溯"的单遍替换，
# 因此 value 不能同时是 key（否则会二次替换）。
# 同义词 -> 标准词。使用"左到右、不回溯"的单遍替换，
# 因此 value 不能同时是 key（否则会二次替换）。
# 只收录"改写后不会把问题带到另一个话题"的映射：
# 比如「多少钱 -> 费用」看着合理，实际会把「学费多少钱」变成「学费费用」，
# 反而匹配不到「学费要交多少啊？」，这类一概不收。
SYNONYM_MAP = {
    # 疑问方式
    "如何": "怎么", "怎样": "怎么", "咋样": "怎么", "咋办": "怎么办", "咋": "怎么",
    "什么时候": "几点", "啥时候": "几点", "什么时间": "几点", "何时": "几点",
    # 金额问法：只归一到"多少"，不动"费用"之类名词，
    # 否则「学费多少钱」会变成「学费费用」反而匹配不上「学费要交多少啊？」
    "多少钱": "多少", "多钱": "多少", "几个钱": "多少",
    # 办事
    "手续": "流程", "遗失": "丢失", "弄丢": "丢失", "丢了": "丢失", "挂失": "补办",
    # 场所 / 机构
    "寝室": "宿舍", "校内": "学校", "本校": "学校", "贵校": "学校", "咱们学校": "学校",
    # 证件（统一到库里使用的说法）
    "一卡通": "校园卡", "借书证": "校园卡", "图书卡": "校园卡",
    # 学业
    "分数": "成绩", "绩点": "成绩",
    # 资金
    "助学贷款": "贷款", "借钱": "贷款",
    # 时间
    "几号": "日期",
}

# 归一化时需要剥离的标点/空白（保留"？"由 tokenize 处理）
_PUNCT_RE = re.compile(r"[\s　!\"#$%&'()*+,\-./:;<=>@\[\\\]^_`{|}~"
                       r"，。、；：（）【】《》—…·！？“”‘’]+")

_FULLWIDTH_MAP = {ord(c): ord(c) - 0xFEE0 for c in
                  "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
                  "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"}

# 长词优先的匹配模式，配合 re.sub 实现"单遍、不回溯"的替换
_SYNONYM_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(SYNONYM_MAP, key=len, reverse=True))
)


def normalize(text: str) -> str:
    """query 归一化：清洗 -> 全角转半角 -> 同义词映射。

    归一化后的字符串同时用于 Redis 精确匹配的 key 和 BM25 检索，
    保证「如何申请奖学金」与库里的「怎么申请奖学金？」落到同一个表示。
    """
    if not text:
        return ""

    text = text.strip().translate(_FULLWIDTH_MAP)
    text = _PUNCT_RE.sub("", text)

    # 单遍左到右替换：长词优先，且替换结果不会被再次扫描
    return _SYNONYM_RE.sub(lambda m: SYNONYM_MAP[m.group(0)], text).strip()


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------

def tokenize(text: str, drop_stopwords: bool = True,
             drop_question_words: bool = False,
             normalize_first: bool = True) -> List[str]:
    """jieba 分词。

    默认先做归一化：分词和归一化必须成对使用，否则同义词两侧表示不一致
    （「学费多少钱」切成 学费/多少钱，「学费要交多少啊」切成 学费/要交/多少，
    覆盖率会算成 1/2 而不是 1/1）。放在这里默认执行，避免调用方漏掉。

    保留词频（不去重）——BM25 依赖 tf，去重会让 BM25 退化成二值模型。
    """
    if not text:
        return []

    if normalize_first:
        text = normalize(text)
    if not text:
        return []

    load_domain_words()

    stopwords = STOPWORDS if drop_stopwords else set()
    if drop_question_words:
        stopwords = set(stopwords) | QUESTION_WORDS

    tokens = []
    for token in jieba.cut(text):
        token = token.strip()
        if not token or token in stopwords:
            continue
        if _PUNCT_RE.fullmatch(token):
            continue
        tokens.append(token)
    return tokens


def keyword_overlap(query: str, target: str, drop_question_words: bool = True) -> float:
    """query 词被 target 覆盖的比例（用集合算，衡量"问的是不是同一件事"）。"""
    query_tokens = set(tokenize(query, drop_question_words=drop_question_words))
    if not query_tokens:
        return 0.0
    target_tokens = set(tokenize(target, drop_question_words=drop_question_words))
    return len(query_tokens & target_tokens) / len(query_tokens)


def char_ngram_similarity(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram Jaccard 相似度。

    用来给 BM25 分数打平的候选做 tiebreak。例如「缓考怎么申请？」和
    「可以申请缓考吗？」分词后 token 集合完全相同，BM25 无法区分，
    但字符 n-gram 能把字面更接近的那个排到前面。
    """
    if not a or not b:
        return 0.0
    a_grams = {a[i:i + n] for i in range(max(len(a) - n + 1, 1))}
    b_grams = {b[i:i + n] for i in range(max(len(b) - n + 1, 1))}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / len(a_grams | b_grams)
