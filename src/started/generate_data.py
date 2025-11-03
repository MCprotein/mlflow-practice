from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_apple_sales_data_with_promo_adjustment(
    base_demand: int = 1000, n_rows: int = 5000
):
    """
    Generates a synthetic dataset for predicting apple sales demand with seasonality
    and inflation.

    This function creates a pandas DataFrame with features relevant to apple sales.
    The features include date, average_temperature, rainfall, weekend flag, holiday flag,
    promotional flag, price_per_kg, and the previous day's demand. The target variable,
    'demand', is generated based on a combination of these features with some added noise.

    Args:
        base_demand (int, optional): Base demand for apples. Defaults to 1000.
        n_rows (int, optional): Number of rows (days) of data to generate. Defaults to 5000.

    Returns:
        pd.DataFrame: DataFrame with features and target variable for apple sales prediction.

    Example:
        >>> df = generate_apple_sales_data_with_seasonality(base_demand=1200, n_rows=6000)
        >>> df.head()
    """

    # 재현을 위한 시드 설정
    np.random.seed(9999)

    # 기간 설정
    dates = [
        datetime.now() - timedelta(days=i) for i in range(n_rows)
    ]
    dates.reverse()

    # feature 생성
    df = pd.DataFrame(
        {
            "date": dates,
            "average_temperature": np.random.uniform(10, 35, n_rows),
            "rainfall": np.random.exponential(scale=5, size=n_rows),
            # True, False -> 1, 0 변환
            "weekend": [(date.weekday() >= 5) * 1 for date in dates],
            "holiday": np.random.choice(
                [0, 1], size=n_rows, p=[0.97, 0.03]
            ),
            "price_per_kg": np.random.uniform(0.5, 3, n_rows),
            "month": [date.month for date in dates],
        }
    )

    df["inflation_multiplier"] = (
        1 + (df.date.dt.year - df.date.dt.year.min()) * 0.03
    )

    df["harvest_effect"] = np.sin(
        2 * np.pi * (df.month - 3) / 12
    ) + np.sin(2 * np.pi * (df.month - 9) / 12)

    df["price_per_kg"] = df.price_per_kg - df.harvest_effect * 0.5

    peak_months = [4, 10]

    df["promo"] = np.where(
        df.month.isin(peak_months),
        1,
        np.random.choice([0, 1], size=n_rows, p=[0.85, 0.15]),
    )

    base_price_effect = -df.price_per_kg * 50
    seasonality_effect = df["harvest_effect"] * 50
    promo_effect = df.promo * 200

    df["demand"] = (
        base_demand
        + seasonality_effect
        + promo_effect
        + df.weekend * 300
        + np.random.normal(0, 50, n_rows)
    ) * df.inflation_multiplier

    df["previous_days_demand"] = df.demand.shift(1)
    # df["previous_days_demand"].fillna(method="bfill", inplace=True)
    df["previous_days_demand"].bfill(inplace=True)

    # 임시 컬럼 제거
    df.drop(
        columns=["inflation_multiplier", "harvest_effect", "month"],
        inplace=True,
    )

    return df
