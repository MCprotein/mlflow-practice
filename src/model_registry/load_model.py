import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import make_regression


def main():
    model_name = "sk-learn-random-forest-reg-model"
    model_verision = "latest"

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_verision}"
    model = mlflow.sklearn.load_model(model_uri)
    print(model, "model")

    # Generate a new dataset for prediction and predict
    x_new, _ = make_regression(
        n_features=4, n_informative=2, random_state=0, shuffle=False
    )
    y_new = model.predict(x_new)
    print(y_new, "y_new")


def load_by_alias():
    client = MlflowClient(tracking_uri="http://localhost:8088")
    model_name = "sk-learn-random-forest-reg-model"
    model_alias = "twothree"
    # 모델의 alias 설정
    model = client.set_registered_model_alias(
        model_name, model_alias, "2"
    )
    model = client.set_registered_model_tag(
        model_name, "created_by", "your_name"
    )

    # model 정보 조회
    model_info = client.get_model_version_by_alias(
        model_name, model_alias
    )
    model_tags = model_info.tags
    print(model_tags, "model_tags")

    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.sklearn.load_model(model_uri)
    print(model, "model")


if __name__ == "__main__":
    with mlflow.start_run(run_name="sklearn-model-load") as run:
        mlflow.set_tracking_uri("http://localhost:8088")
        # main()
        load_by_alias()
