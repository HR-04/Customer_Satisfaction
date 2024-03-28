from logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstarct Class for all Models

    """
    @abstractmethod
    def train(self,x_train,y_train):
        """
        Abstract method to train the model

        Args:
            x_train : Training Data
            y_train : Training Labels
            
        """
        pass
    

class LinearRegressionModel( Model):
    """
        Linear Regression Model
    """
    def train(self,x_train,y_train,**kwargs):
        """
        Trains the linear regression

        Args:
            x_train : Training Data
            y_train : Training Labels
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train,y_train)
            logging.info("Linear regression model trained")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
    
        
        