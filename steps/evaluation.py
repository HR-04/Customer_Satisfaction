import logging
import pandas as pd
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

from zenml import step

from src.evaluation import MSE,R2,RMSE


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin,
                   x_test:pd.DataFrame,
                   y_test:pd.DataFrame,
                   )->Tuple[
                       Annotated[float,'r2_score'],
                       Annotated[float,'rmse'],
                   ]:
    
    """
    Evaluates the model on the ingested data.
    
    Args:
    df: The Ingested Data
    """
    try:
        prediction = model.predict(x_test)
        mse_class = MSE()
        mse=mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("mse",mse)
        
        r2_class = R2()
        r2_score=r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("r2_score",r2_score)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("rmse",rmse)
        
        return r2_score,rmse
    except Exception as e:
        logging.error("Error in Evaluating the model:{}".format(e))
        raise e
    
    