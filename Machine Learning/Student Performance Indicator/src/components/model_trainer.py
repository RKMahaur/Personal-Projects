import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "ElasticNet": ElasticNet(),
                "Support Vector Regressor": SVR(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor()
                }

            params = {
                "Linear Regression":{},
                "Lasso Regression":{
                    'alpha':[10,1,.1,.01,.05,.001]
                    },
                "Ridge Regression":{
                    'alpha':[10,1,.1,.01,.05,.001],
                    'solver':['auto','svd','cholesky','lspr','saga']
                    },
                "ElasticNet":{
                    'alpha':[10,1,.1,.01,.05,.001],
                    'selection':['cyclic','random']
                    },
                "Support Vector Regressor":{
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],       
                    'epsilon': [0.01, 0.1, 0.5],     
                    'gamma': ['scale', 'auto'] 
                    },
                "KNeighbors Regressor":{
                    'n_neighbors': [3, 4, 5, 6],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'metric': ['minkowski', 'euclidean', 'manhattan'] 
                    },
                "Decision Tree Regressor":{
                    'criterion':['gini','entropy','log_loss','squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                    },
                "Random Forest Regressor":{
                    'criterion':['gini','entropy','log_loss','squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                    },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                    },
                "Gradient Boosting Regressor":{
                    'loss':['log_loss','exponential','squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['mse','mae','squared_error', 'friedman_mse','absolute_error'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                    },
                "XGBRegressor":{
                    'booster':['gbtree','gblinear','dart'],
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    'tree_method':['auto','exact','approx','hist','gpu_hist'],
                    }
                }

            model_report:dict=evaluate_models(X_train=X_train,
                                              Y_train=Y_train,
                                              X_test=X_test,
                                              Y_test=Y_test,
                                              models=models,
                                              params=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            Y_test_pred=best_model.predict(X_test)

            r2_square = r2_score(Y_test,Y_test_pred)

            return r2_square
             
        except Exception as e:
            raise CustomException(e,sys)