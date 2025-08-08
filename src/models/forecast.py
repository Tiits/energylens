import pandas as pd
from prophet import Prophet


def fit_prophet(
    df: pd.DataFrame,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.05,
    holidays: pd.DataFrame = None
) -> Prophet:
    """
    Train a Prophet model on the training data.

    Args:
        df (pd.DataFrame): DataFrame with columns ['ds', 'y'] for Prophet.
        weekly_seasonality (bool): enable weekly seasonality.
        daily_seasonality (bool): enable daily seasonality.
        yearly_seasonality (bool): enable yearly seasonality.
        changepoint_prior_scale (float): constraint on the flexibility of the changepoints.
        holidays (pd.DataFrame): DataFrame of holidays for Prophet (optional).

    Returns:
        Prophet: Trained model.
    """
    model = Prophet(
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        holidays=holidays
    )
    model.fit(df)
    return model


def predict_prophet(
    model: Prophet,
    periods: int,
    freq: str = '15min'
) -> pd.DataFrame:
    """
    Generate forecasts from a trained Prophet model.

    Args:
        model (Prophet): Trained Prophet model.
        periods (int): Number of time steps to forecast.
        freq (str): Time frequency (e.g., '15min', '1h').

    Returns:
        pd.DataFrame: DataFrame of forecasts with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper'] and other internal columns.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


def fit_predict_prophet(
    df: pd.DataFrame,
    periods: int,
    freq: str = '15min',
    **model_kwargs
) -> (Prophet, pd.DataFrame):
    """
    Train and predict in a single step.

    Args:
        df (pd.DataFrame): DataFrame for training (columns 'ds', 'y').
        periods (int): Number of prediction steps.
        freq (str): Time frequency.
        model_kwargs: Additional arguments for fit_prophet.

    Returns:
        Tuple[Prophet, pd.DataFrame]: Trained model and forecast DataFrame.
    """
    model = fit_prophet(df, **model_kwargs)
    forecast = predict_prophet(model, periods, freq)
    return model, forecast
