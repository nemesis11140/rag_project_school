"""LLM 回答生成。

上下文构建使用**父块**（parent_text）：召回用子块保证精准，喂给 LLM 用父块
保证上下文完整——这正是父子分块的意义所在。
"""
from typing import Any, Dict, List, Optional

import requests

from config import LLM_CONFIG, SYSTEM_CONFIG
from ..common.assistant_intent import get_assistant_reply

SYSTEM_PROMPT = """你是"小交"，上海交通大学的智能校园问答助手。

你的职责：
1. 为上海交通大学的学生、教职工提供校园事务咨询服务
2. 基于学校官方文档（学生手册、教职工手册等）回答问题
3. 提供准确、专业、友好的服务

回答规范：
1. 严格基于提供的参考信息回答，不编造信息
2. 语气亲切友好，像一位贴心的学长/学姐
3. 回答简洁明了，直接给出答案
4. 如果参考信息中没有相关内容，坦诚告知并建议联系相关部门
5. 重要信息可以用数字编号列出
6. 不要使用Markdown格式符号如#、*、-、>、|等，但必须使用正确的中文标点符号
7. 不要使用emoji表情
8. 如果用户的问题涉及到之前的对话，请结合上下文回答
9. 用户消息开头的[对话信息]包含当前对话轮数等元数据"""

GENERAL_SYSTEM_PROMPT = """你是"小交"，上海交通大学的智能校园问答助手。
用户问的是与校园事务无关的闲聊或通用问题，请友好、简洁地回应，
并自然地引导用户提出校园相关的问题。不要使用Markdown格式符号和emoji。"""


class AnswerGenerator:
    def __init__(self):
        self.api_url = LLM_CONFIG.get("api_url", "")
        self.api_key = LLM_CONFIG.get("api_key", "")
        self.model_name = LLM_CONFIG.get("model_name", "")
        self.temperature = LLM_CONFIG.get("temperature", 0.7)
        self.max_tokens = LLM_CONFIG.get("max_tokens", 1000)
        self.extra_body = LLM_CONFIG.get("extra_body", {}) or {}
        self.context_max_chars = SYSTEM_CONFIG.get("context_max_chars", 4000)

    # ---------- 对外接口 ----------

    def generate_rag(self, query: str, documents: List[Dict[str, Any]],
                     chat_history: Optional[List[Dict]] = None,
                     chat_count: int = 0) -> str:
        if not documents:
            return "抱歉，我在知识库中没有找到与您问题相关的内容。请您补充一下具体信息，或者联系学校相关部门的老师。"

        context = self._build_context(documents)
        if not context.strip():
            return "抱歉，我在知识库中没有找到与您问题相关的内容。请您补充一下具体信息，或者联系学校相关部门的老师。"

        prompt = f"""{self._build_context_info(chat_count)}请根据以下学校官方文档内容，回答用户的问题。

参考文档：
{context}

用户问题：{query}

请基于参考文档内容，用自然、友好的语言回答用户的问题。回答时请注意：
1. 先直接回答用户的核心问题
2. 如果涉及具体流程，用数字编号列出步骤
3. 回答简洁明了，控制在300字以内，不要罗列与问题无关的条款
4. 不要使用Markdown格式符号如#、*、-、>、|等，不要使用emoji
5. 必须使用正确的中文标点符号，确保句子通顺流畅"""

        return self._call_llm(prompt, is_general=False, chat_history=chat_history)

    def generate_general(self, query: str, chat_history: Optional[List[Dict]] = None,
                         chat_count: int = 0) -> str:
        """闲聊/通用问题的兜底回答。"""
        assistant_reply = get_assistant_reply(query)
        if assistant_reply is not None:
            return assistant_reply

        if not self._llm_available():
            return (
                "你好！我是小交，上海交通大学的智能校园问答助手。\n"
                "我可以帮你解答学籍管理、奖学金、宿舍生活、办事流程等校园问题，"
                "请问有什么可以帮你的吗？"
            )

        prompt = f"""{self._build_context_info(chat_count)}用户问题：{query}

请用简洁、友好的方式回应，并提醒用户你可以帮助解答校园事务相关问题。"""

        return self._call_llm(prompt, is_general=True, chat_history=chat_history)

    # ---------- 上下文构建 ----------

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """按父块组装上下文，同一父块只出现一次。"""
        parts: List[str] = []
        total = 0
        seen_parents = set()

        for doc in documents:
            parent_key = (doc.get("document_name", ""), doc.get("parent_index", 0))
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)

            # 优先用父块（上下文完整），没有父块才退回子块
            content = (doc.get("parent_text") or doc.get("child_text") or "").strip()
            if not content:
                continue

            source = doc.get("document_name", "未知文档")
            block = f"来源：{source}\n{content}"

            if total + len(block) > self.context_max_chars:
                remaining = self.context_max_chars - total
                if remaining < 200:
                    break
                block = block[:remaining] + "……"
            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    @staticmethod
    def _build_context_info(chat_count: int) -> str:
        if chat_count > 0:
            return f"[对话信息：当前是第{chat_count + 1}轮对话，用户之前已经问了{chat_count}个问题]\n\n"
        return "[对话信息：当前是第1轮对话]\n\n"

    # ---------- LLM 调用 ----------

    def _llm_available(self) -> bool:
        return bool(self.api_url and self.api_key)

    def _call_llm(self, prompt: str, is_general: bool = False,
                  chat_history: Optional[List[Dict]] = None, timeout: int = 60) -> str:
        if not self._llm_available():
            print("[AnswerGenerator] LLM 未配置，返回降级话术")
            if is_general:
                return "你好！我是小交，上海交通大学的智能校园问答助手。请问有什么可以帮你的吗？"
            return "抱歉，当前无法访问大模型服务。请您联系学校相关部门获取准确信息。"

        messages = [{
            "role": "system",
            "content": GENERAL_SYSTEM_PROMPT if is_general else SYSTEM_PROMPT
        }]
        # 保留最近 12 条消息（约 6 轮），与 API 接收上限一致。
        for msg in (chat_history or [])[-12:]:
            role, content = msg.get("role"), msg.get("content")
            if role and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        # 透传服务商特有参数（如 {"thinking": {"type": "disabled"}} 关闭思考过程）
        body.update(self.extra_body)

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=body,
                timeout=timeout
            )

            if response.status_code == 200:
                payload = response.json()
                choice = (payload.get("choices") or [{}])[0]
                answer = choice.get("message", {}).get("content", "")

                # 推理模型（如 deepseek-flash）会把思考过程放在 reasoning_content 里，
                # 并一起计入 max_tokens。额度不够时 content 直接为空或被截断，
                # 这里必须显式告警，否则表现为"答非所问/答一半"却查不出原因
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    reasoning_tokens = (payload.get("usage", {})
                                        .get("completion_tokens_details", {})
                                        .get("reasoning_tokens"))
                    print(
                        f"[AnswerGenerator] 警告：输出被 max_tokens({self.max_tokens}) 截断"
                        f"（推理消耗 {reasoning_tokens} tokens），建议调大 max_tokens"
                    )
                if not (answer or "").strip():
                    print(
                        f"[AnswerGenerator] 警告：content 为空（finish_reason={finish_reason}），"
                        "该模型可能把内容都放进了 reasoning_content"
                    )
                    return "抱歉，这次没能生成有效回答，请重试或换个问法。"

                return self._clean_answer(answer)

            # 把错误码原样打出来：像 AllocationQuota.FreeTierOnly 这类 code
            # 才是根因所在，截断消息体很容易误判成"余额不足"
            print(f"[AnswerGenerator] LLM API error: HTTP {response.status_code} {self._error_code(response)}")
            return self._friendly_error(response)

        except requests.exceptions.Timeout:
            print("[AnswerGenerator] LLM API timeout")
            return "抱歉，回答生成超时了，请稍后重试或联系学校相关部门获取帮助。"
        except Exception as e:
            print(f"[AnswerGenerator] LLM API error: {type(e).__name__}: {e}")
            return "抱歉，回答生成服务暂时不可用，请稍后重试或联系学校相关部门。"

    @staticmethod
    def _error_code(response) -> str:
        try:
            error = response.json().get("error", {})
            return f"[{error.get('code') or error.get('type')}] {error.get('message', '')}"
        except Exception:
            return response.text[:200]

    @staticmethod
    def _friendly_error(response) -> str:
        """按错误码给出可操作的提示，而不是笼统的"服务不可用"。"""
        code = ""
        try:
            code = response.json().get("error", {}).get("code", "") or ""
        except Exception:
            pass

        if "FreeTierOnly" in code:
            return (
                "抱歉，大模型服务当前不可用：账户开启了「仅使用免费额度」模式，"
                "免费额度已用完。请在阿里云百炼控制台关闭该模式后重试。"
            )
        if "InvalidApiKey" in code or "Authentication" in code or response.status_code == 401:
            return "抱歉，大模型服务鉴权失败（API Key 无效）。请检查 .env 中的 LLM_API_KEY。"
        if "Throttling" in code or response.status_code == 429:
            return "抱歉，请求过于频繁，请稍后重试。"
        return "抱歉，回答生成服务暂时不可用，请稍后重试或联系学校相关部门。"

    @staticmethod
    def _clean_answer(answer: str) -> str:
        import re

        answer = re.sub(r"#{1,6}\s*", "", answer)
        answer = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", answer)
        answer = re.sub(r"^[-*]\s+", "", answer, flags=re.MULTILINE)
        answer = re.sub(r"^>\s+", "", answer, flags=re.MULTILINE)
        answer = re.sub(r"\|", " ", answer)
        answer = re.sub(r"```.*?```", "", answer, flags=re.DOTALL)
        answer = re.sub(r"`([^`]+)`", r"\1", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer.strip()
