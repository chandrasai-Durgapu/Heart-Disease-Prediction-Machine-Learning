import os
import pandas as pd
from utils.logger import get_logger
import time
logger=get_logger()

def ingest_data()->pd.DataFrame:
    try: 
        start_time=time.perf_counter()
        logger.info("Data Ingestion begins")

        if not os.path.exists("artifacts/ingestion"):
            os.makedirs("artifacts/ingestion", exist_ok=True)

        df=pd.read_csv('data/heart.csv')
        df.to_csv("artifacts/ingestion/ingested.csv", index=False)
        logger.info("Ingested data saved ")
        end_time=time.perf_counter()
        logger.info(f"Data Ingestion completed and total time taken is {end_time-start_time:.2f} seconds")
        return df
    except Exception as e:
        logger.error(e)
        


if __name__=="__main__":
    ingest_data()