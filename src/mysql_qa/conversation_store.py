"""MySQL 对话记录存储。

会话通过 user_id 关联登录账号；会话列表、标题和消息正文全部保存在 MySQL。
每次操作单独建立连接，避免 FastAPI 线程池并发复用同一个 PyMySQL 连接。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

import pymysql

from config import DB_NAME, MYSQL_CONFIG


class ConversationStore:
    conversations_table = "chat_conversations"
    messages_table = "chat_messages"

    def _connect(self):
        return pymysql.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            charset=MYSQL_CONFIG["charset"],
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor,
        )

    @staticmethod
    def _serialize_time(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat(timespec="seconds") if value else None

    @classmethod
    def _serialize_conversation(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "message_count": int(row.get("message_count", 0)),
            "created_at": cls._serialize_time(row.get("created_at")),
            "updated_at": cls._serialize_time(row.get("updated_at")),
        }

    def ensure_tables(self) -> None:
        """创建会话与消息表；可重复调用。"""
        # chat_conversations 的外键依赖 users 表。
        from src.auth import AuthManager
        AuthManager().ensure_tables()

        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.conversations_table} (
                        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT UNSIGNED NOT NULL,
                        title VARCHAR(200) NOT NULL DEFAULT '新对话',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_user_updated (user_id, updated_at),
                        CONSTRAINT fk_conversation_user FOREIGN KEY (user_id)
                            REFERENCES users(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                      COLLATE=utf8mb4_unicode_ci
                """)

                # 从匿名 Cookie 版本平滑迁移：保留旧 owner_id 数据但不再读取，
                # 新会话全部写 user_id。不能自动归属旧记录，以免泄露给错误账号。
                cursor.execute(f"SHOW COLUMNS FROM {self.conversations_table}")
                columns = {row["Field"] for row in cursor.fetchall()}
                if "user_id" not in columns:
                    cursor.execute(f"""
                        ALTER TABLE {self.conversations_table}
                        ADD COLUMN user_id BIGINT UNSIGNED NULL AFTER id
                    """)
                if "owner_id" in columns:
                    cursor.execute(f"""
                        ALTER TABLE {self.conversations_table}
                        MODIFY COLUMN owner_id CHAR(36) NULL
                    """)

                cursor.execute(f"""
                    SELECT COUNT(*) AS total
                    FROM information_schema.statistics
                    WHERE table_schema = %s AND table_name = %s
                      AND index_name = 'idx_user_updated'
                """, (DB_NAME, self.conversations_table))
                if cursor.fetchone()["total"] == 0:
                    cursor.execute(f"""
                        ALTER TABLE {self.conversations_table}
                        ADD INDEX idx_user_updated (user_id, updated_at)
                    """)

                cursor.execute(f"""
                    SELECT COUNT(*) AS total
                    FROM information_schema.table_constraints
                    WHERE constraint_schema = %s AND table_name = %s
                      AND constraint_name = 'fk_conversation_user'
                """, (DB_NAME, self.conversations_table))
                if cursor.fetchone()["total"] == 0:
                    cursor.execute(f"""
                        ALTER TABLE {self.conversations_table}
                        ADD CONSTRAINT fk_conversation_user FOREIGN KEY (user_id)
                            REFERENCES users(id) ON DELETE CASCADE
                    """)
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.messages_table} (
                        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        conversation_id BIGINT UNSIGNED NOT NULL,
                        role ENUM('user', 'assistant') NOT NULL,
                        content MEDIUMTEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_conversation_id (conversation_id, id),
                        CONSTRAINT fk_chat_message_conversation
                            FOREIGN KEY (conversation_id)
                            REFERENCES {self.conversations_table}(id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                      COLLATE=utf8mb4_unicode_ci
                """)
            conn.commit()
        finally:
            conn.close()

    def list_conversations(self, user_id: int, limit: int = 30) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT c.id, c.title, c.created_at, c.updated_at,
                           COUNT(m.id) AS message_count
                    FROM {self.conversations_table} c
                    LEFT JOIN {self.messages_table} m ON m.conversation_id = c.id
                    WHERE c.user_id = %s
                    GROUP BY c.id, c.title, c.created_at, c.updated_at
                    ORDER BY c.updated_at DESC, c.id DESC
                    LIMIT %s
                """, (user_id, limit))
                return [self._serialize_conversation(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def create_conversation(self, user_id: int, title: str = "新对话") -> Dict[str, Any]:
        title = (title or "新对话").strip()[:200] or "新对话"
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO {self.conversations_table} (user_id, title) VALUES (%s, %s)",
                    (user_id, title),
                )
                conversation_id = cursor.lastrowid
            conn.commit()
            return self.get_conversation(user_id, conversation_id)
        finally:
            conn.close()

    def get_conversation(self, user_id: int,
                         conversation_id: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT c.id, c.title, c.created_at, c.updated_at,
                           COUNT(m.id) AS message_count
                    FROM {self.conversations_table} c
                    LEFT JOIN {self.messages_table} m ON m.conversation_id = c.id
                    WHERE c.id = %s AND c.user_id = %s
                    GROUP BY c.id, c.title, c.created_at, c.updated_at
                """, (conversation_id, user_id))
                row = cursor.fetchone()
                return self._serialize_conversation(row) if row else None
        finally:
            conn.close()

    def get_messages(self, user_id: int, conversation_id: int,
                     limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        if not self.get_conversation(user_id, conversation_id):
            return None

        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT recent.id, recent.role, recent.content, recent.created_at
                    FROM (
                        SELECT m.id, m.role, m.content, m.created_at
                        FROM {self.messages_table} m
                        WHERE m.conversation_id = %s
                        ORDER BY m.id DESC
                        LIMIT %s
                    ) AS recent
                    ORDER BY recent.id ASC
                """, (conversation_id, limit))
                return [{
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": self._serialize_time(row["created_at"]),
                } for row in cursor.fetchall()]
        finally:
            conn.close()

    def append_exchange(self, user_id: int, conversation_id: int,
                        query: str, answer: str) -> Optional[Dict[str, Any]]:
        """在同一事务中写入一轮用户问题和助手回答，并自动生成标题。"""
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT title FROM {self.conversations_table}
                    WHERE id = %s AND user_id = %s
                    FOR UPDATE
                """, (conversation_id, user_id))
                row = cursor.fetchone()
                if not row:
                    conn.rollback()
                    return None

                cursor.executemany(f"""
                    INSERT INTO {self.messages_table} (conversation_id, role, content)
                    VALUES (%s, %s, %s)
                """, [
                    (conversation_id, "user", query),
                    (conversation_id, "assistant", answer),
                ])

                title = row["title"]
                if title == "新对话":
                    compact_query = " ".join(query.split())
                    title = compact_query[:18] + ("…" if len(compact_query) > 18 else "")
                cursor.execute(f"""
                    UPDATE {self.conversations_table}
                    SET title = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND user_id = %s
                """, (title or "新对话", conversation_id, user_id))
            conn.commit()
            return self.get_conversation(user_id, conversation_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def delete_conversation(self, user_id: int, conversation_id: int) -> bool:
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    DELETE FROM {self.conversations_table}
                    WHERE id = %s AND user_id = %s
                """, (conversation_id, user_id))
                deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
        finally:
            conn.close()
