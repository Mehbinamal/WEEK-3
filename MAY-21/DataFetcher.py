import pandas as pd
def fetch_csv_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {file_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The CSV file is empty")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
