import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

#importing the custom modules
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass
@dataclass
class DataTransformationConfig:
    preprocessed_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    #method that creates the data transformer object
    def get_data_transformer_object(self):
        try:
            numerical_columns= ["writing score","reading score"]
            categorical_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

            #creating the pipeline for the numerical columns
            numerical_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler(with_mean=False))
                ]
            )


            #creating the pipeline for the categorical columns
            categorical_pipeline = Pipeline(
                steps=[      
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder()),
                ('std_scaler', StandardScaler(with_mean=False))
                ]
            )

            #logging the information
            logging.info("Created the pipeline for the numerical and categorical columns")
            logging.info("Numerical column standard scaling is  completed")
            logging.info("Categorical column encoding completed")


            #creating the column transformer object
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline, numerical_columns),
                    ("categorical_pipeline",categorical_pipeline, categorical_columns)

                ]
                )
              

            return preprocessor
            

        except Exception as e:
            raise CustomException(e,sys)

    

    #method that initiates the data transformation
    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("Reading the train and test data completed")
            logging.info("Obtaining the preprocessing object")

            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="math score"
            numerical_columns= ["writing score","reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying the preprocessing object on the train and test data")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving the preprocessor object")


            #saving the pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessed_obj_file_path,
                obj=preprocessor_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessed_obj_file_path)


        except Exception as e:
            raise CustomException(e,sys)