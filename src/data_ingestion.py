import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_function import read_yaml
from google.cloud import storage

logger = get_logger(__name__)



class DataIngestion:
    
    def __init__(self,config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        
        os.makedirs(RAW_DIR ,exist_ok=True)
        
        logger.info(f"Data Ingestion Start With {self.bucket_name} And File Is {self.file_name}")
        
        
    def download_csv_from_gcp(self):
        
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            
            blob.download_to_filename(RAW_FILE_PATH)
            
            logger.info(f" CSV File Is Successfully Download To {RAW_FILE_PATH}")
            
        except Exception as e:
            logger.error("Error While Downloading The CSV File ")
            raise CustomException("Failed To Download The CSV File", sys)
        
        
    def split_data(self):
        try:
            
            logger.info("Starting The Splitting Data")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data , test_data = train_test_split(data , test_size=1-self.train_test_ratio, random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            
            logger.info(f"Train Data Saved To {TRAIN_FILE_PATH}")
            logger.info(f"Test Data Saved To {TEST_FILE_PATH}")
            
        except Exception as e:
            logger.error("Error While Splitting The Data")
            raise CustomException("Failed To Split Train And Test Data", sys)
        
    def run(self):
        try:
            
            logger.info("Starting The DataIngestion Process")
            
            self.download_csv_from_gcp()
            self.split_data()
            
            logger.info("Data Ingestion Completed Successfully")
            
        except Exception as e:
            logger.error(f"CustomException : {str(e)}")
            raise e
            
        finally:
            logger.info("DataIngestion Completed")
            

if __name__=="__main__":

    data_ingestion = DataIngestion(read_yaml(CONFIG_FILE)) 
    data_ingestion.run()
            