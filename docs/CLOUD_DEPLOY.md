# 云端部署与升级

本次修复覆盖：高频问答误匹配、四张兼容空表阻止恢复、MySQL 已恢复但 Milvus 缺失、数据库故障被显示成模型加载失败，以及修复后必须重启才能重新预热的问题。

应用镜像通过 Docker Hub 和阿里 ACR 发布。数据库和模型独立持久化，镜像不携带账号、聊天记录、备份、密钥或模型权重。升级应用不需要再次覆盖现有业务数据库。

## 首次部署或补齐缺失组件

1. 将发布包里的 `docker-compose.yml`、`deploy.sh`、`.env.example` 放到同一个项目目录。
2. 保留已有 `.env`；首次部署则复制示例填写 `MYSQL_PASSWORD`、`REDIS_PASSWORD`、`LLM_API_KEY`。
3. 保留原来的 Compose 项目名和数据卷。上传完整模型到 `model/bge-m3`、`model/bge-reranker-base`；`bert_intent` 可选。
4. 将完整备份文件夹上传到 `backups/`。
5. 选择镜像并运行统一部署脚本。

```bash
# Docker Hub
export APP_IMAGE=zcong11140/rag_project_school:20261015-fix1

# 或使用杭州阿里 ACR
# export APP_IMAGE=crpi-3h7l0sjo9afn3xet.cn-hangzhou.personal.cr.aliyuncs.com/zzc11140/rag_project_school:20261015-fix1

bash deploy.sh backups/rag-project-school-db-backup-20261012-151748
```

脚本先拉取应用镜像、停止旧应用，启动数据库并等待健康，然后调用新镜像中的恢复工具。工具复用结构兼容的空表，补建缺失表；已有业务数据的整个组件会保留，缺失的其他组件仍可恢复。它不会把当前账号、聊天或向量数据覆盖成旧备份。

恢复时核对 SHA-256、表结构与全部导入内容，保留每个组件的状态输出。仅当整个目标 MySQL 库的表都为空，且结构差异属于可识别的历史认证表字段时，才允许按备份结构重建对应空表；存在业务记录时不会重建。原文备份不会被修改。恢复工具支持 `--components milvus` 等参数，只连接指定组件。

恢复后检查 Milvus 项目库、集合和数据量，以及必需模型文件。检查通过才启动应用，并等待模型预热和健康检查成功；任何一步失败都会停止，不会输出“部署完成”。日志里 `existing_kept` 代表该组件保留现有业务数据，并不表示重新导入了备份。

## 已完成数据部署，只升级应用

```bash
export APP_IMAGE=zcong11140/rag_project_school:20261015-fix1
bash deploy.sh
```

不传备份目录时不会执行数据恢复。保持同一个项目目录、Compose 项目名和卷名。不要使用 `docker compose down -v` 升级应用。

## 不更新部署文件时的手动操作

```bash
docker compose pull app
docker compose stop app
docker compose run --rm --no-deps -T app python scripts/migrate_to_docker.py restore \
  --backup-dir /app/backups/rag-project-school-db-backup-20261012-151748 \
  --use-configured-target --resume --keep-existing
docker compose run --rm --no-deps -T app python scripts/check_deployment.py
docker compose up -d --no-deps --no-build --wait --wait-timeout 600 app
```

已有完整数据时跳过恢复命令；缺少 Milvus 时也可以给恢复命令添加 `--components milvus`。旧备份自带的 `restore.sh` 和 `restore_databases.py` 是旧版工具，更新镜像不会更新已上传的备份文件，所以本次请使用上面的统一部署脚本或镜像内的新工具。

## 故障定位

```bash
curl -sS http://127.0.0.1:5000/api/health
docker compose logs --no-color --tail=80 app
```

健康接口返回 `problem.code`、`problem.message` 和 `problem.hint`。缺少知识库时明确显示需要恢复的库或集合；缺少模型时提示模型目录；MySQL、Milvus 连接失败各自报告。健康检查要求预热成功，不会把空知识库标记成就绪。修复依赖后可刷新页面触发重试预热。

不把数据库异常归为模型故障，也不为了通过健康检查自动创建空知识库。首次没有备份时，先按照 README 初始化数据与构建文档知识库，再运行不带备份参数的部署脚本。
