from generate_data import (
    generate_apple_sales_data_with_promo_adjustment,
)
from tracking_mlflow import tracking_mlflow


def main():
    # client = MlflowClient(tracking_uri="http://localhost:8088")

    # all_experiments = client.search_experiments()
    # print(all_experiments, "all_experiments")

    # default_experiment = [
    #     {
    #         "name": experiment.name,
    #         "lifecycle_stage": experiment.lifecycle_stage,
    #     }
    #     for experiment in all_experiments
    #     if experiment.name == "Default"
    # ][0]

    # pprint.pprint(default_experiment)

    # experiment_description = (
    #     "This is the grocery forecasting project. "
    #     "This experiment contains the produce models for apples."
    # )

    # experiment_tags = {
    #     "project_name": "grocery-forecasting",
    #     "store_dept": "produce",
    #     "team": "stores-ml",
    #     "project_quarter": "Q3-2023",
    #     "mlflow.note.content": experiment_description,
    # }

    # produce_apples_experiment = client.create_experiment(
    #     name="Apple_Models", tags=experiment_tags
    # )

    # 사용자 정의 태그 이름은 백틱으로 감싼다.

    # apples_experiment = client.search_experiments(
    #     filter_string="tags.`project_name` = 'grocery-forecasting'"
    # )
    # print(vars(apples_experiment[0]))

    data = generate_apple_sales_data_with_promo_adjustment(
        base_demand=1_000, n_rows=1_000
    )
    print(data[-20:])

    tracking_mlflow(data)


if __name__ == "__main__":
    print("Hello from mlflow-practice!")
    main()
    print("Done")
