import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path : Path of the data_path
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data from the data_path.
        """
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    
    """
    Ingest data from a data_path.
    
    Args:
        data_path: Path to the data file.
    Returns:
        pd.DataFrame: The ingested data.
    """
    
    try:
        ingestor = IngestData(data_path)
        df = ingestor.get_data()
        return df  
    except Exception as e:
        logging.error(f'Error ingesting data: {str(e)}')
        raise e
