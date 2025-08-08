import pandas as pd
import sklearn.datasets 
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class CaliforniaHousingLoader:
    def load(self) -> pd.DataFrame:
        # Loads the California Housing dataset as a pandas DataFrame.

        logging.info("Loading California housing data.")

        dataset = sklearn.datasets.fetch_california_housing(as_frame=True)
        df = dataset.frame
        # The original target column name is 'MedHouseVal'. We'll rename it to 'price'.
        if 'MedHouseVal' in df.columns:
            df = df.rename(columns={'MedHouseVal': 'price'})

        logging.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        return df
    
class AmesHousingLoader:
    DATA_PATH = BASE_DIR / "src/data/raw/AmesHousing.csv"

    def load(self) -> pd.DataFrame:
        logging.info("Loading Ames housing data.")
        try:
            df_raw = pd.read_csv(self.DATA_PATH)
        except FileNotFoundError:
            logging.error(f"Ames Housing Data not found at: {self.DATA_PATH}")
            raise

        df = self._clean_data(df_raw)

        logging.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and performs initial cleaning."""
        # Standardize column names: lowercase, replace spaces/special chars with _
        cols = df.columns
        new_cols = []
        for col in cols:
            new_col = col.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            new_cols.append(new_col)
        df.columns = new_cols

        # The target column in the raw CSV is 'saleprice' after cleaning
        if "saleprice" in df.columns:
            df = df.rename(columns={"saleprice": "price"})

        return df