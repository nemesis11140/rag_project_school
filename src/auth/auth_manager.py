"""MySQL 用户认证与服务端登录会话。"""
import base64
import hashlib
import os
import re
import secrets
from typing import Any, Dict, Optional

import pymysql

from config import DB_NAME, MYSQL_CONFIG


_USERNAME_RE = re.compile(r"^[\w\u4e00-\u9fff]{3,32}$", re.UNICODE)
_PBKDF2_ITERATIONS = 310_000


class AuthManager:
    users_table = "users"
    sessions_table = "user_sessions"

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

    def ensure_tables(self) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.users_table} (
                        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(32) NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_users_username (username)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                      COLLATE=utf8mb4_unicode_ci
                """)
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.sessions_table} (
                        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT UNSIGNED NOT NULL,
                        token_hash CHAR(64) NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_sessions_token_hash (token_hash),
                        INDEX idx_sessions_user (user_id),
                        INDEX idx_sessions_expiry (expires_at),
                        CONSTRAINT fk_session_user FOREIGN KEY (user_id)
                            REFERENCES {self.users_table}(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                      COLLATE=utf8mb4_unicode_ci
                """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def validate_credentials(username: str, password: str) -> tuple[str, str]:
        username = (username or "").strip()
        password = password or ""
        if not _USERNAME_RE.fullmatch(username):
            raise ValueError("用户名需为 3～32 位中文、字母、数字或下划线")
        if len(password) < 8:
            raise ValueError("密码至少需要 8 个字符")
        if len(password) > 128:
            raise ValueError("密码不能超过 128 个字符")
        return username, password

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
        )
        return "pbkdf2_sha256${}${}${}".format(
            _PBKDF2_ITERATIONS,
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        )

    @staticmethod
    def _verify_password(password: str, encoded: str) -> bool:
        try:
            algorithm, iterations, salt_b64, digest_b64 = encoded.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            salt = base64.b64decode(salt_b64)
            expected = base64.b64decode(digest_b64)
            actual = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, int(iterations)
            )
            return secrets.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def register(self, username: str, password: str) -> Dict[str, Any]:
        username, password = self.validate_credentials(username, password)
        password_hash = self._hash_password(password)
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(f"""
                        INSERT INTO {self.users_table} (username, password_hash)
                        VALUES (%s, %s)
                    """, (username, password_hash))
                except pymysql.err.IntegrityError as exc:
                    raise ValueError("用户名已存在") from exc
                user_id = cursor.lastrowid
            conn.commit()
            return {"id": user_id, "username": username}
        finally:
            conn.close()

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        username = (username or "").strip()
        password = password or ""
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id, username, password_hash
                    FROM {self.users_table}
                    WHERE username = %s
                    LIMIT 1
                """, (username,))
                row = cursor.fetchone()
                if not row or not self._verify_password(password, row["password_hash"]):
                    return None
                return {"id": row["id"], "username": row["username"]}
        finally:
            conn.close()

    def reset_password(self, username: str, new_password: str) -> bool:
        """演示用找回密码：只核对账户存在性，更新密码并使所有旧会话失效。"""
        username, new_password = self.validate_credentials(username, new_password)
        password_hash = self._hash_password(new_password)
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id FROM {self.users_table}
                    WHERE username = %s LIMIT 1 FOR UPDATE
                """, (username,))
                user = cursor.fetchone()
                if not user:
                    conn.rollback()
                    return False
                cursor.execute(
                    f"UPDATE {self.users_table} SET password_hash = %s WHERE id = %s",
                    (password_hash, user["id"]),
                )
                cursor.execute(
                    f"DELETE FROM {self.sessions_table} WHERE user_id = %s",
                    (user["id"],),
                )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_session(self, user_id: int, days: int = 7) -> str:
        token = secrets.token_urlsafe(32)
        token_hash = self._token_hash(token)
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.sessions_table} WHERE expires_at <= CURRENT_TIMESTAMP"
                )
                cursor.execute(f"""
                    INSERT INTO {self.sessions_table} (user_id, token_hash, expires_at)
                    VALUES (%s, %s, DATE_ADD(CURRENT_TIMESTAMP, INTERVAL %s DAY))
                """, (user_id, token_hash, days))
            conn.commit()
            return token
        finally:
            conn.close()

    def get_user_by_token(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        token_hash = self._token_hash(token)
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT u.id, u.username
                    FROM {self.sessions_table} s
                    JOIN {self.users_table} u ON u.id = s.user_id
                    WHERE s.token_hash = %s AND s.expires_at > CURRENT_TIMESTAMP
                    LIMIT 1
                """, (token_hash,))
                row = cursor.fetchone()
                return {"id": row["id"], "username": row["username"]} if row else None
        finally:
            conn.close()

    def delete_session(self, token: Optional[str]) -> None:
        if not token:
            return
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.sessions_table} WHERE token_hash = %s",
                    (self._token_hash(token),),
                )
            conn.commit()
        finally:
            conn.close()

    def delete_user(self, user_id: int) -> bool:
        """删除指定用户；主要用于测试清理，关联会话由外键级联删除。"""
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.users_table} WHERE id = %s", (user_id,)
                )
                deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
        finally:
            conn.close()
