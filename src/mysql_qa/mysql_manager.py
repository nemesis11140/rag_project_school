import pymysql
from typing import Optional, Dict, Any, List
from config import MYSQL_CONFIG, DB_NAME, TABLE_NAME


class MySQLManager:
    def __init__(self):
        self.host = MYSQL_CONFIG["host"]
        self.port = MYSQL_CONFIG["port"]
        self.user = MYSQL_CONFIG["user"]
        self.password = MYSQL_CONFIG["password"]
        self.charset = MYSQL_CONFIG["charset"]
        self.db_name = DB_NAME
        self.table_name = TABLE_NAME
        self.connection = None

    def connect(self):
        if self.connection is None or not self._is_connection_alive():
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                charset=self.charset,
                database=self.db_name
            )
        return self.connection

    def _is_connection_alive(self):
        if self.connection is None:
            return False
        try:
            self.connection.ping(reconnect=True)
            return True
        except Exception:
            return False

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def get_answer(self, question: str) -> Optional[str]:
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(f"SELECT answer FROM {self.table_name} WHERE question = %s LIMIT 1", (question,))
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
        except Exception as e:
            print(f"[MySQLManager] Error getting answer: {e}")
            return None

    def get_all_questions(self) -> List[str]:
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(f"SELECT question FROM {self.table_name}")
            results = cursor.fetchall()
            cursor.close()
            return [row[0] for row in results]
        except Exception as e:
            print(f"[MySQLManager] Error getting all questions: {e}")
            return []

    def get_all_qa_pairs(self) -> List[Dict[str, Any]]:
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(f"SELECT id, question, answer FROM {self.table_name}")
            results = cursor.fetchall()
            cursor.close()
            return [{"id": row[0], "question": row[1], "answer": row[2]} for row in results]
        except Exception as e:
            print(f"[MySQLManager] Error getting all QA pairs: {e}")
            return []

    def get_answer_by_id(self, qa_id: int) -> Optional[str]:
        """按主键取答案——比按 question 文本精确匹配更可靠。"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(f"SELECT answer FROM {self.table_name} WHERE id = %s", (qa_id,))
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
        except Exception as e:
            print(f"[MySQLManager] Error getting answer by id: {e}")
            return None

    def insert_qa_pairs(self, pairs: List[tuple]) -> int:
        """批量写入，单事务提交。"""
        if not pairs:
            return 0
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.executemany(
                f"INSERT INTO {self.table_name} (question, answer) VALUES (%s, %s)",
                pairs
            )
            conn.commit()
            count = cursor.rowcount
            cursor.close()
            return count
        except Exception as e:
            print(f"[MySQLManager] Error batch inserting QA pairs: {e}")
            return 0

    def truncate_qa_pairs(self) -> bool:
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(f"TRUNCATE TABLE {self.table_name}")
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"[MySQLManager] Error truncating table: {e}")
            return False

    def insert_qa_pair(self, question: str, answer: str) -> bool:
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO {self.table_name} (question, answer) VALUES (%s, %s)",
                (question, answer)
            )
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"[MySQLManager] Error inserting QA pair: {e}")
            return False

    def create_table_if_not_exists(self):
        try:
            conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                charset=self.charset
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")
            cursor.execute(f"USE {self.db_name}")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_question (question(100))
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            conn.commit()
            cursor.close()
            conn.close()
            print(f"[MySQLManager] Table {self.table_name} is ready")
        except Exception as e:
            print(f"[MySQLManager] Error creating table: {e}")