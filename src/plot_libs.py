import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

def plot_hist(data: pd.DataFrame, feature_name=None, class_labels = 'label'):
    """
    Plot a histogram of the features in the dataset.
    """
    hist_data = []
    group_labels = []
    for class_ in data[class_labels].unique():
        hist_data.append(data.loc[data[class_labels] == class_][feature_name])
        group_labels.append("Class %d" % class_)

    fig = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=True)
    
    return fig


def plot_explain(pcamodel):
    """ plot pca explain for a given pcamodel
    """
    fig, ax = plt.subplots(figsize=(7,2))
    plt.bar(range(1,len(pcamodel.explained_variance_ )+1),pcamodel.explained_variance_ )
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1,len(pcamodel.explained_variance_ )+1),
            np.cumsum(pcamodel.explained_variance_),
            c='red',
            label="Cumulative Explained Variance")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)

    return fig, ax