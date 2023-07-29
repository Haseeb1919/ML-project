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


    def initiate_model_trainer(self,data_array):
        try:
            logging.info("Initiating the model training process")
            logging.info("Splitting the data into train and test")
            
            # Split data into features (X) and target (y)
            X = data_array[:, :-1]
            y = data_array[:, -1]
            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




            # Verify the shapes of train_array and test_array
            print("Shape of X_train:", X_train.shape)
            print("Shape of y_train:", y_train.shape)
            print("Shape of X_test:", X_test.shape)
            print("Shape of y_test:", y_test.shape)



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
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]
            
            #making a threshold for the best model
            if best_model_score< 0.8:
                raise CustomException("No best best model found")
            logging.info(f"Best found model on both training and testing dataset")
            # logging.info(f"Best model is {best_model}")

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
 