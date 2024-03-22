from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run#1
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from azureml.core import Workspace, Experiment

ws = Workspace.from_config()
run = Run.get_context()


def clean_data(data):
    #imputer = SimpleImputer(strategy="most_frequent")
    #data.iloc[:, :] = imputer.fit_transform(data)

    target_transformer = ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), [0])]
    )
    
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )
    
    features_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, make_column_selector(dtype_exclude="object")),
            ("categorical", categorical_transformer, ["island"]),
        ]
    )
    
    temp_df = data.to_pandas_dataframe()
    temp_df.dropna(inplace = True)
    y_df = target_transformer.fit_transform(np.array(temp_df.species.values).reshape(-1, 1))    
    temp_df = temp_df.drop('species', axis=1)
    x_df = features_transformer.fit_transform(temp_df)
    return x_df,y_df


dataset = ws.datasets["penguins"] 
df = dataset.to_pandas_dataframe()

ds = Dataset.from_pandas_dataframe(df)

x, y = clean_data(ds)

# TODO: Split data into train and test sets.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=403,shuffle=True)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('-f')
    
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength.")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    primary_metric_name="Accuracy"
    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    joblib.dump(model,'outputs/model.joblib')
    accuracy = model.score(x_test, y_test)
    
    run.log("Accuracy", np.float(accuracy))
if __name__ == '__main__':
    main()