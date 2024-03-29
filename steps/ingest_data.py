import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv("./data/olist_customers_dataset.csv",index_col=0,parse_dates=True)
        return df
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    
    """
    Ingest data from a data_path.
    
    Args:
        data_path: Path to the data file.
    Returns:
        pd.DataFrame: The ingested data.
    """
    
    try:
        ingestor = IngestData()
        df = ingestor.get_data()
        return df  
    except Exception as e:
        logging.error(f'Error ingesting data: {str(e)}')
        raise e
