# Milvus 缺少项目数据库时的单独恢复

适用情形：应用提示 `database not found[database=rag_project_school]`，检查发现 Milvus 只有空的 `default` 库。此时只补恢复向量库，不重新导入 MySQL 和 Redis。

将 `scripts/restore_milvus_only.py` 上传到云服务器项目根目录，文件名保持 `restore_milvus_only.py`。完整备份文件夹仍位于项目的 `backups/rag-project-school-db-backup-20261012-151748/` 下。

在云服务器项目根目录执行：

```bash
docker compose stop app

docker compose run --rm --no-deps -T -e PYTHONPATH=/app \
  --volume "$PWD/restore_milvus_only.py:/app/restore_milvus_only.py:ro" \
  app python /app/restore_milvus_only.py \
  --backup-dir /app/backups/rag-project-school-db-backup-20261012-151748
```

脚本通过一次性应用容器连接 Compose 的 Milvus 服务，检查快照 SHA-256、库名和集合结构，再恢复集合、索引、文本与现有向量。不加载模型，不重新计算向量，不连接 MySQL 或 Redis。

本备份预期输出：

```text
集合 rag_knowledge_base：676 条，恢复并校验成功
Milvus 恢复并校验完成，数据库：rag_project_school。未连接 MySQL 或 Redis。
```

已有集合只有在结构和全部内容均与备份一致时才会跳过；遇到不同内容会停止，保留现有数据。不要删除数据卷或重新执行完整数据库恢复来绕过检查。

确认显示“Milvus 恢复并校验完成”后，再启动原有应用容器：

```bash
docker compose start app
```

应用会重新预热。稍后检查：

```bash
curl -sS http://127.0.0.1:5000/api/health
```

`models_ready` 为 `true` 表示模型就绪。如果仍失败，查看最新的应用异常；恢复向量库只能解决数据库缺失这一处原因。
