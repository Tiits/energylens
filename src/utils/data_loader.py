import pandas as pd

def load_heapo_raw(path: str) -> pd.DataFrame:
    """
    Load the raw HEAPO CSV and return a pandas DataFrame.

    Args:
        path (str): Path to the raw HEAPO CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'household', 'consumption']
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])
    # Minimal check
    expected_cols = {'timestamp', 'household_id', 'consumption' }
    if not expected_cols.issubset(df.columns):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Colonnes manquantes dans le CSV HEAPO: {missing}")
    # Rename column household_id for consistency
    df = df.rename(columns={'household_id': 'household'})
    return df


def preprocess_heapo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the HEAPO data for modeling:
    - Sort by timestamp
    - Remove missing values
    - Ensure a continuous 15-minute time index
    - Rename to ds (date) and y (consumption) for Prophet

    Args:
        df (pd.DataFrame): Raw HEAPO DataFrame

    Returns:
        pd.DataFrame: DataFrame ready for forecasting
    """
    # Select useful columns
    df = df[['timestamp', 'household', 'consumption']].copy()
    # Sort
    df = df.sort_values('timestamp')
    # Drop NA
    df = df.dropna(subset=['consumption'])
    # Resample to ensure a regular 15-minute frequency
    df = (
        df
        .set_index('timestamp')
        .groupby('household')
        .resample('15T')['consumption']
        .mean()
        .reset_index()
    )
    # Rename for Prophet
    df = df.rename(columns={'timestamp': 'ds', 'consumption': 'y'})
    # ds must be datetime
    df['ds'] = pd.to_datetime(df['ds'])
    return df
