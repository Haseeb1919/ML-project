import os 
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)



def evaluate_models(X_train,y_train,X_test,y_test ,models):
    try:
        report = {}
        #loop through the models in the dictionary
        for i in range(len(list(models))):
            model=list(models.values())[i]
            #fitting the model
            model.fit(X_train,y_train)
            #predictions
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)


            #metrics
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

    

            #saving the metrics in the dictionary
            report[list(models.keys())[i]] = test_model_score

        return report    



    except Exception as e:
        raise CustomException(e,sys)