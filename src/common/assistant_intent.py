"""识别明确询问助手身份、能力的表达，避免与校园事务的用途问题混淆。"""
import re
import unicodedata
from typing import Optional


ASSISTANT_INTRO = (
    "你好！我是小交，校园问答助手。"
    "我可以根据校园知识库，帮你查询学籍管理、奖助学金、选课考试、"
    "宿舍生活、校园卡和办事流程等信息，也支持结合当前对话回答追问。"
    "例如，你可以问“奖学金怎么申请？”或“学生证丢了怎么办？”。"
    "我提供信息咨询，不能代你办理业务；知识库没有的内容会说明。"
)

_SUBJECT = r"(?:你|您|小交|(?:这个|这款|本)?(?:校园)?(?:问答|智能)?(?:助手|机器人|系统))"
_IDENTITY = re.compile(
    rf"(?:{_SUBJECT}(?:是谁|叫什么(?:名字)?|是什么(?:助手|机器人|系统)?)"
    r"|(?:介绍|介绍一下)(?:你|您|小交|自己|你自己)|自我介绍(?:一下)?)"
)
_CAPABILITIES = re.compile(
    rf"{_SUBJECT}(?:都|到底)?(?:"
    r"有(?:什么|哪些|啥)(?:用|功能|作用|本领|能力)"
    r"|是(?:做什么|干什么|干啥|干嘛)的"
    r"|(?:能|可以)(?:帮我|帮我们)?(?:做|干)(?:什么|啥)(?:事|事情)?"
    r"|(?:能|可以)(?:帮我|帮我们)?(?:回答|解答)(?:什么|哪些|啥)(?:校园|学校)?问题"
    r"|(?:能|可以)(?:帮我|帮我们)(?:什么|哪些|啥)"
    r"|的(?:功能|作用|能力)(?:是什么|有哪些|有哪些内容))"
)


def get_assistant_reply(query: str) -> Optional[str]:
    """完整句式匹配；“你知道校园卡有什么用吗”仍属于校园问题。"""
    text = unicodedata.normalize("NFKC", query or "").strip()
    text = re.sub(r"[\s，。！？、,.!?：:；;‘’“”\"']", "", text)
    if text in {"你好", "您好", "嗨", "hello", "hi", "小交你好", "你好小交"}:
        return ASSISTANT_INTRO
    text = re.sub(r"^(?:(?:请问|请|麻烦|你好|您好|我想知道|想问一下|告诉我))+", "", text)
    text = re.sub(r"[吗呢呀啊吧]+$", "", text)
    if _IDENTITY.fullmatch(text) or _CAPABILITIES.fullmatch(text):
        return ASSISTANT_INTRO
    return None
