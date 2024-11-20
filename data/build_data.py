import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

def build_dataset(no_class: int = 2, no_features: int = 3, no_samples: int = 1000):
    """
    Build a dataset with (no_samples) samples and (no_features) features with number of classes determined by input no_class.
    Input parameters:
        no_class (int): Number of classes in the dataset. Default is 2.
        no_features (int): Number of features in the dataset. Default is 3.
        no_samples (int): Number of samples in the dataset. Default is 1000.
    Returns:
        data (pd.DataFrame): DataFrame output.
    """
    # Generate random features
    data = pd.DataFrame([])
    label = pd.DataFrame([])
    for i in range(no_class):
        random_miu = np.random.random_integers(10)
        random_sigma = 0.5
        random_data = pd.DataFrame([])
        for j in range(no_features):
            random_data = pd.concat([random_data, pd.DataFrame(np.random.normal(random_miu, random_sigma, int(no_samples/no_class)), columns=[f'F{j}'])], axis=1)
        random_data['label'] = [i]*(int(1000/no_class))
        data = pd.concat([data, random_data], axis=0)
    return data

    
def build_iris_data():
    """
    Build a dataset of Iris flower samples.
    Returns:
    data (pd.DataFrame): DataFrame output.
    """
    dataset = px.data.iris()
    dataset = dataset[["sepal_width", "sepal_length", "petal_width", "petal_length", "species"]].copy()
    return dataset