from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == '__main__':
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    training_pipeline(data_path= "F:/Customer_Satisfaction/Data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:C:\Users\ADMIN\AppData\Roaming\zenml\local_stores\de7d4237-565b-439f-b188-4fc5e141c1ba\mlruns"