import numpy as np
import pandas as pd
import streamlit as st

def build_dataset(no_class: int = 2):
    """
    Build a dataset with 1000 samples and 10 features with number of classes determined by input no_class.
    Input parameters:
    no_class (int): Number of classes in the dataset. Default is 2.
    Returns:
    X (pd.DataFrame): DataFrame containing the features of the dataset.
    """
    # Generate random features
    data = pd.DataFrame([])
    label = pd.DataFrame([])
    for i in range(no_class):
        random_miu = np.random.random_integers(10)
        random_sigma = 0.5
        random_data = pd.DataFrame([])
        for j in range(10):
            random_data = pd.concat([random_data, pd.DataFrame(np.random.normal(random_miu, random_sigma, int(1000/no_class)), columns=[f'F{j}'])], axis=1)
        random_data['label'] = [i]*(int(1000/no_class))
        data = pd.concat([data, random_data], axis=0)
    return data

    
    