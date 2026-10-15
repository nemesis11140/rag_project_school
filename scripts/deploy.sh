#!/usr/bin/env bash
set -euo pipefail
# 用法：bash deploy.sh [backups/备份目录]；可选 APP_IMAGE 指定 ACR 镜像。
if [[ ! -f docker-compose.yml && ! -f compose.yml ]]; then
    echo "请在项目 Compose 目录执行。" >&2
    exit 1
fi
if [[ ! -f .env ]]; then
    echo "缺少 .env，请先填写数据库密码和 LLM_API_KEY。" >&2
    exit 1
fi
if [[ $# -gt 1 ]]; then
    echo "用法：bash deploy.sh [backups/备份目录]" >&2
    exit 1
fi
backup_dir="${1:-}"
if [[ -n "$backup_dir" ]]; then
    backup_dir="${backup_dir%/}"
    case "$backup_dir" in
        backups/*) ;;
        *) echo "备份目录必须位于当前项目 backups/ 下。" >&2; exit 1 ;;
    esac
    if [[ "$backup_dir" == *".."* || ! -f "$backup_dir/snapshot.json" || ! -f "$backup_dir/snapshot.sha256" ]]; then
        echo "备份路径或文件不完整，未执行部署。" >&2
        exit 1
    fi
fi
docker compose config --quiet
docker compose pull app
docker compose stop app
docker compose up -d --wait --wait-timeout 240 mysql redis milvus
if [[ -n "$backup_dir" ]]; then
    docker compose run --rm --no-deps -T app python scripts/migrate_to_docker.py restore \
        --backup-dir "/app/$backup_dir" --use-configured-target --resume --keep-existing
fi
docker compose run --rm --no-deps -T app python scripts/check_deployment.py
docker compose up -d --no-deps --no-build --wait --wait-timeout 600 app
docker compose exec -T app python -c \
    'import json,urllib.request; s=json.load(urllib.request.urlopen("http://127.0.0.1:5000/api/health")); print(json.dumps(s,ensure_ascii=False,indent=2)); assert s["models_ready"]'
echo "部署完成，数据库与模型均已就绪。"
