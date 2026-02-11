import os 
import pandas as pd 
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_function import read_yaml,load_data
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


logger = get_logger(__name__)

class DataProcessor:
    
    def __init__(self,train_path,test_path,processed_dir,config_path):
        
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        
    
    def process_data(self,df):
        
        try:
            
            logger.info("Starting Our Data Processing")
            
            logger.info("Filling The Missing Value Of Age & Embarked Column")
        
            df['Age'] = df['Age'].fillna(df['Age'].mean())
            df['Embarked'] = df['Embarked'].fillna('S')
            
            logger.info("Applying The mapping to Embarked & Sex Columm")
            
            df['Embarked'] = df['Embarked'].map({'C': 0,'Q': 1,'S': 2})
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
            
            logger.info("Creating The New Feature HasCabin & FamilySize")
            
            df['HasCabin'] = df['Cabin'].notnull().astype(int)
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            
            logger.info("Droping The Column")
            
            df.drop(columns=['Cabin','PassengerId','Ticket','Name'],inplace=True)
            
            return df
            
        except Exception as e:
            logger.error("Error During The Processed Data {e}")
            raise CustomException("Failed While Processed Data", e)
        
    def balanced_data(self,df):
        try:
            
            logger.info("Balancing The Data")
            
            X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'HasCabin']]
            y = df['Survived']
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                    X_train, y_train
                )
            
            balanced_df = pd.DataFrame(X_train_resampled , columns=X.columns)
            balanced_df["Survived"] = y_train_resampled
            
            logger.info("Data balanced sucesffuly")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during balancing data step {e}")
            raise CustomException("Error while balancing data", e)
        
    def save_data(self,df,file_path):
        try:
            
            logger.info("Saving Our Data In Processed Dir")
            
            df.to_csv(file_path, index=False)
            
            logger.info(f"Data Saved Succesfully At {file_path}")
            
        except Exception as e:
            logger.error("Error While Saving The Data {e}")
            raise CustomException("Error While Saving The Data ", e)
    
    
    def process(self):
        
        try:
            
            logger.info("Loading Data From Raw Directory")
            
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            
            train_df = self.process_data(train_df)
            test_df = self.process_data(test_df)
            
            train_df = self.balanced_data(train_df)
            test_df = self.balanced_data(test_df)
            
            self.save_data(train_df, PROCESSED_TRAIN_FILE_PATH)
            self.save_data(test_df,PROCESSED_TEST_FILE_PATH)
            
            logger.info("Data Processing Completed Succesfully")
        
        except Exception as e:
            logger.error("Error During Processing Pipeling Step {e}")
            raise CustomException("Error While Data Preprocessing Pipeline",  e)
        
        
if __name__=="__main__":
    
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_FILE)
    processor.process()