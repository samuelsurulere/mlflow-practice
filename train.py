import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow


def evaluation_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2


def train_model(data_path, alpha, l1_ratio):
    # Read the wine-quality csv file from the URL
    data = pd.read_csv(data_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Set default values if no alpha is provided
    if float(alpha) is None:
        alpha = 0.5

    # Set default values if no l1_ratio is provided
    if float(l1_ratio) is None:
        l1_ratio = 0.5

    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = evaluation_metrics(test_y, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")


if __name__ == "__main__":
    # train.py is the entry point for training the model
    data_path = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
    )
    train_model(data_path, 0.3, 0.7)