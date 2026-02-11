from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_function import read_yaml
from config.path_config import *


if __name__=="__main__":
    
    data_ingestion = DataIngestion(read_yaml(CONFIG_FILE))
    data_ingestion.run()

    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_FILE)
    processor.process()

    trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH,PROCESSED_TEST_FILE_PATH,MODEL_OUTPUT_PATH)
    trainer.run()