#!/bin/bash
set -e

# .env.sh 파일 로드
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env.sh" ]; then
    source "$SCRIPT_DIR/.env.sh"
    echo "✅ Loaded environment variables from .env.sh"
else
    echo "❌ Warning: .env.sh not found. Using default values."
fi

# 환경 변수 확인
echo "Using AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_DEFAULT_REGION"
echo "Backend Store: $MLFLOW_BACKEND_STORE_URI"
echo "Artifact Root: $MLFLOW_ARTIFACT_ROOT"

# MLflow 서버 실행
uv run mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT"