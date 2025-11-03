import mlflow
import mlflow.pytorch  # type: ignore
import pandas as pd
import torch
import torch.nn as nn
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.entities import Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Helper function to prepare data
def prepare_data(df: pd.DataFrame):
    x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    return x, y


# Helper function to compute accuracy
def compute_accuracy(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
    return accuracy


# Define a basic PyTorch classifier
class IrisClassifier(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    # Set MLflow tracking URI to local file system
    mlflow.set_tracking_uri("http://localhost:8088")

    # Load Iris dataset and prepare the DataFrame
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target

    train_df, test_df = train_test_split(
        iris_df, test_size=0.2, random_state=42
    )

    # Prepare training data
    train_dataset = mlflow.data.from_pandas(train_df, name="train")
    X_train, y_train = prepare_data(train_dataset.df)

    # Define the PyTorch model and move it to the device
    input_size = X_train.shape[1]
    hidden_size = 16
    output_size = len(iris.target_names)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    scripted_model = IrisClassifier(
        input_size, hidden_size, output_size
    ).to(device)
    scripted_model = torch.jit.script(scripted_model)

    # Start a run to represent the training job
    with mlflow.start_run() as run:
        # Load the training dataset with MLflow. We will link training metrics to this dataset.
        train_dataset: Dataset = mlflow.data.from_pandas(
            train_df, name="train"
        )
        X_train, y_train = prepare_data(train_dataset.df)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            scripted_model.parameters(), lr=0.01
        )

        for epoch in range(101):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            out = scripted_model(X_train)
            loss = criterion(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log a checkpoint with metrics every 10 epochs
            if epoch % 10 == 0:
                # Each newly created LoggedModel checkpoint is linked with its name and step
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=scripted_model,
                    name="torch-iris",  # step으로 구분
                    step=epoch,
                    input_example=X_train.numpy(),
                )
                # log params to the run, LoggedModel inherits those params
                mlflow.log_params(
                    params={
                        "n_layers": 3,
                        "activation": "ReLU",
                        "criterion": "CrossEntropyLoss",
                        "optimizer": "Adam",
                    }
                )
                # Log metric on training dataset at step and link to LoggedModel
                mlflow.log_metric(
                    key="accuracy",
                    value=compute_accuracy(
                        scripted_model, X_train, y_train
                    ),
                    step=epoch,
                    model_id=model_info.model_id,
                    dataset=train_dataset,
                )


if __name__ == "__main__":
    main()
