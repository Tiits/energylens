import os
import requests
import pandas as pd
import zipfile
import glob

DOWNLOAD_URL = "https://zenodo.org/records/15056919/files/heapo_data.zip?download=1"  # cite arXiv:2503.16993

def load_heapo_raw(zip_path: str, download_url: str = DOWNLOAD_URL) -> pd.DataFrame:
    """
    Download (if needed), extract and load the HEAPO 15-minute CSV data into a DataFrame.

    Args:
        zip_path (str): Path to the HEAPO zip archive.
    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'household', 'consumption'].
    """
    # Download the ZIP if missing
    if not os.path.exists(zip_path):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract if not already extracted
    extract_dir = os.path.splitext(zip_path)[0]
    if not os.path.isdir(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(os.path.dirname(zip_path))

    # Path to 15-minute data
    data_folder = os.path.join(extract_dir, "smart_meter_data", "15min")
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    # Read and combine all household files
    df_list = []
    for file in csv_files:
        df_i = pd.read_csv(file, parse_dates=["timestamp"])
        household_id = os.path.splitext(os.path.basename(file))[0]
        df_i["household"] = household_id
        df_list.append(df_i[["timestamp", "household", "consumption"]])

    df = pd.concat(df_list, ignore_index=True)

    # Minimal column sanity check
    expected_cols = {"timestamp", "household", "consumption"}
    if not expected_cols.issubset(df.columns):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Missing columns after loading HEAPO data: {missing}")

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
