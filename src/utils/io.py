import os
import pandas as pd
from typing import List, Optional, Dict
from src.constant import RAW_FEATURES, PROCESSED_FEATURES

def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except Exception as e:
        raise IOError(f"Error reading CSV file at {path}: {e}")
    

def read_multi_csv(paths: List[str], **kwargs) -> pd.DataFrame:
    """Đọc nhiều file CSV và gộp thành một DataFrame."""
    dfs = []
    for p in paths:
        df = read_csv_safe(p, **kwargs)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def save_csv_safe(df, path, **kwargs):
    """
    Lưu DataFrame thành file CSV. Tự động tạo các thư mục nếu chưa tồn tại.
    Nếu file đã tồn tại thì sẽ được ghi đè.
    """
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    try:
        df.to_csv(path, index=False, **kwargs)  # Luôn ghi đè nếu file đã có!
    except Exception as e:
        raise IOError(f"Error saving CSV file at {path}: {e}")
    



