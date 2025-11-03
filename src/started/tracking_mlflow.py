import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


def tracking_mlflow(data: pd.DataFrame):
    mlflow.set_tracking_uri("http://localhost:8088")

    # 현재 활성화된 실험 설정
    apple_experiment = mlflow.set_experiment("Apple_Models")

    # run 이름 설정
    # 이름 설정 안하면 랜덤 고유 값 자동 생성
    run_name = "apples_rf_test"

    # 모델이 저장될 artifact path 설정
    artifact_path = "rf_apples"

    x = data.drop(columns=["date", "demand"])
    y = data["demand"]

    (
        x_train,
        x_val,
        y_train,
        y_val,
    ) = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 888,
    }

    # RandomForestRegressor 모델 생성
    rf = RandomForestRegressor(**params)

    # 모델 학습
    rf.fit(x_train, y_train)

    # 검증 셋 예측
    y_pred = rf.predict(x_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    # 모델 입출력 스키마 정의 (정수 컬럼 경고 방지)
    from mlflow.models import infer_signature

    signature = infer_signature(x_train, y_train)

    with mlflow.start_run(run_name=run_name) as run:
        # 모델 학습에 사용된 파라미터 로깅
        mlflow.log_params(params)
        # 밸리데이션 중에 발생한 에러 메트릭 로깅
        mlflow.log_metrics(metrics)
        # 나중에 사용할 학습 모델 인스턴스 로깅
        mlflow.sklearn.log_model(
            sk_model=rf,
            name=artifact_path,
            signature=signature,
            input_example=x_val,
            # Model Registry에 자동 등록 (버전 자동 증가)
            registered_model_name="apple_sales_model",
        )
