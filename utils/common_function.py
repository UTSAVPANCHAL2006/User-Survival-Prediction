import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        
        if not os.path.exists(file_path):
            raise FileNotFoundError ("File Not Found In Given File Path")
        
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully Load Yaml File")
            return config
        
    except Exception as e:
        logger.error("Error While Reading The Yaml File")
        raise CustomException ("Failed to Load Yaml File", e)
    
def load_data(path):
    try:
        logger.info("Loading The Data")
        return pd.read_csv(path)
    except Exception as e :
        logger.error(f"Error Loading The Data {e}")
        raise CustomException ("Failed To Load Data", e)