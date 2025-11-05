import os

import mlflow
import mlflow.sklearn  # type: ignore
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# .env 파일 로드
load_dotenv()


def register_model():
    x, y = make_regression(
        n_features=4, n_informative=2, random_state=0, shuffle=False
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    params = {"max_depth": 2, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)

    y_pred = model.predict(x_test)
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        name="sklearn-model",
        input_example=x_train,
        registered_model_name="sk-learn-random-forest-reg-model",
        # tags 추가해도 모델 등록 시 적용되지 않음
        tags={
            "run_id": run.info.run_id,
            "created_by": "your_name",
            "alias": "sklearn-model",
        },
        signature=infer_signature(x_train, y_train),
    )
    print(mlflow.get_artifact_uri())


if __name__ == "__main__":
    # Tracking URI를 start_run() 전에 설정!
    mlflow.set_tracking_uri("http://localhost:8088")

    # S3를 사용하는 실험 설정 (없으면 자동 생성, 서버의 --default-artifact-root 사용)
    mlflow.set_experiment("production-models")

    with mlflow.start_run(
        run_name="sklearn-model-registration"
    ) as run:
        register_model()
