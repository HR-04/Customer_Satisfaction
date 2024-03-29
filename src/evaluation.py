import logging
from abc import ABC,abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):
    """
    Abstract Class Defining for Evaluation our model class
    """
    
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        calculate the scores for the model
        
        args:
            y_true:True Labels
            y_Pred : Predicted labels
        returns: 
            None
        """
        pass
    
class MSE(Evaluation):
    """
    Evaluation Stratergy that uses Mean Squared Error
    """
    
    def calaculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculate the scores for the model
        
        args:
            y_true:True Labels
            y_Pred : Predicted labels
        returns: 
            None
        """
        try:
            mse = mean_squared_error(y_true,y_pred)
            logging.info("Mean Squared Error : {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE:{}".format(e))
            raise e
        
class R2(Evaluation):
    """
    Evaluation Stratergy that uses Root Mean Square Error
    """
    def calaculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculate the scores for the model
        
        args:
            y_true:True Labels
            y_Pred : Predicted labels
        returns: 
            None
        """
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true,y_pred)
            logging.info("R2 Score : {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score:{}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    Evalauation Stratergy That uses Root Mean Squared Error
    """
    def calaculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculate the scores for the model
        
        args:
            y_true:True Labels
            y_Pred : Predicted labels
        returns: 
            None
        """
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true,y_pred))
            logging.info("RMSE : {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE:{}".format(e))
            raise e
    