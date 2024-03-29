import numpy as np
import pandas as pd
from materializer.custom_materializer import cs_materializer
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integration.mlflow.model_deployers.mlflow_model_deployer import(MLFloweModelDeployer,)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParamaters,Output

from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=True,settings={"docker_settings":docker_settings})
def continuous_deployment_pipeline(
    min_accuracy:float=0.92,
    workers:int=1,
    timeout:int=DEFAULT_SERVICE_START_STOP_TIMEOUT,

):
    df = ingest_df()
    X_train,X_test,y_train,y_test = clean_data(df)
    model = train_model(X_train,X_test,y_train,y_test)
    r2_score,rmse = evaluate_model(model,X_test,y_test)
    