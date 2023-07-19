#train different models and figuring out the best model

#system libraries
import os 
import sys
from dataclasses import dataclass

#model libraries
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#metrics libraries
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#custopm libraries
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Initiating the model training process")
            logging.info("Splitting the data into train and test")
            X_train, X_test, y_train, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models = {
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "CatBoostRegressor": CatBoostRegressor(),
                "XGBRegressor": XGBRegressor()
            }

            model_report:dict=evaluate_models(
                models=models,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test)

            
            #finding the best model
            best_model_name = max(sorted(model_report.key()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_name)]

            best_model= models[best_model_name]
            
            #making a threshold for the best model
            if best_model_score< 0.8:
                raise CustomException("The best model is not performing well",sys)

            logging.info(f"Best model is {best_model}")

            #save the best model

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            
            logging.info("Best model saved")

            #checking the model performance
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e,sys)
