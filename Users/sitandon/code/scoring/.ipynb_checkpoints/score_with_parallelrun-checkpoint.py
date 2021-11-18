import pickle
import json
import numpy
from sklearn.linear_model import Ridge
from azureml.core.model import Model
import pandas as pd


def init():
    global model
    import joblib

    # load the model from file into a global object
    model_path = Model.get_model_path(model_name="diabetes")
    model = joblib.load(model_path)


def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []
    for file in mini_batch:
        
        df = pd.read_csv(file)
        x_columns = ["age", "gender", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
        y_columns = ["Y"]
        df.columns = x_columns + y_columns

        result = model.predict(df[x_columns])
        # prepare each image
        resultList.append(result)
    return resultList