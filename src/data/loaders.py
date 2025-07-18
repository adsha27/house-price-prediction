import pandas as pd
import sklearn.datasets 
import logging

class CaliforniaHousingLoader:
    def load(self) -> pd.DataFrame:
        """
        Loads the California Housing dataset as a pandas DataFrame.

        Uses scikit-learn's fetch_california_housing and returns it as a frame,
        renaming the target column to 'price' for consistency.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        logging.info("Loading California housing data.")

        # fetch_california_housing with as_frame=True is the modern, preferred way.
        # It directly returns a Bunch object with a .frame attribute (the DataFrame).
        dataset = sklearn.datasets.fetch_california_housing(as_frame=True)

        # The DataFrame includes features and the target.
        df = dataset.frame

        # The original target column name is 'MedHouseVal'. We'll rename it to 'price'.
        # This makes our pipeline's target column name consistent across different datasets.
        if 'MedHouseVal' in df.columns:
            df = df.rename(columns={'MedHouseVal': 'price'})

        logging.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        return df