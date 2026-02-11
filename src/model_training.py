import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from utils.common_function import read_yaml,load_data
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from xgboost import XGBClassifier


logger = get_logger(__name__)

class ModelTraining:
    
    def __init__(self,train_path,test_path,model_output_path):
        
        self.train_path = train_path
        self.test_path = test_path
        
        self.model_output_path = model_output_path 
        
        
        
    def load_and_split_data(self): 
        
        try:
            
            logger.info(f"Loading Data From The {self.train_path}")      
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading Data From The {self.test_path}")      
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=['Survived'])
            y_train = train_df['Survived']
            
            X_test = test_df.drop(columns=['Survived'])
            y_test = test_df['Survived']
            
            logger.info("Data Sucessfully Spiltted For Model Training")
            
            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            logger.error("While Spilting The Data ")
            raise CustomException("Error While Spilting The Data", sys)
        
    def train_xgb(self, X_train, y_train):
        
        try:
            
            logger.info("Intializing The Model ")
            
            model = XGBClassifier()
            
            logger.info("Training The Model")
            
            model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            logger.error("Error while training model")
            raise CustomException("Failed to train model" ,  sys)   
        
    def evaluate_model(self , model , X_test ,y_test):
        try:
            
            logger.info("Evaluating The Model")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")
            
            return {
                "accuracy" : accuracy,
                "precison" : precision,
                "recall" : recall,
                "f1" : f1
            }
        except Exception as e:
            logger.error("Error while evaluating model ")
            raise CustomException("Failed to evaluate model" , sys)
        
        
    def save_model(self,model):
                
        try:
            
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            
            logger.info("Saving The Model")
            
            joblib.dump(model , self.model_output_path)
            
            logger.info(f"Model Saved At {self.model_output_path}")
            
        except Exception as e:
            logger.error("Error While Saving The Model ")
            raise CustomException("Failed While Saving The Model", sys)
        
        
    def run(self):
    
        try:
            
            X_train, y_train, X_test, y_test = self.load_and_split_data()
            model = self.train_xgb(X_train, y_train)
            self.evaluate_model(model, X_test, y_test)
            self.save_model(model)
            
            logger.info("Model training pipeline completed successfully")

        except Exception as e:
            logger.error("Error in model training pipeline")
            raise CustomException("Model training pipeline failed", sys)

        
        
        
if __name__=="__main__":
    
    trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH,PROCESSED_TEST_FILE_PATH,MODEL_OUTPUT_PATH)
    trainer.run()