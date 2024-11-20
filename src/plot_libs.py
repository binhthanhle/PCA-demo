import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.figure_factory as ff
import plotly.express as px

def create_hist_plot(data: pd.DataFrame, feature_name=None, class_labels = 'label'):
    """
    Create a plot a histogram of the features in the dataset.
    """
    hist_data = []
    group_labels = []
    for class_ in data[class_labels].unique():
        hist_data.append(data.loc[data[class_labels] == class_][feature_name])
        group_labels.append(f"Class {class_}")

    fig = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=True)
    
    return fig


def hist_plot(dataset: pd.DataFrame, class_labels: str = None):
    """
    Plot a histogram of the features in the dataset.
    """
    if class_labels is None:
        class_labels = dataset.columns[-1]

    feature = st.sidebar.selectbox("Choose a feature", dataset.columns[:-1].to_list())
    fig = create_hist_plot(dataset,feature_name=feature, class_labels = class_labels)
    st.plotly_chart(fig, use_container_width=True)



def plot_explain(data: pd.DataFrame, pca, num_components: int = None):
    """Plot the explanation of a given PCA data"""

    # Explained variance plot
    if num_components is None:
        num_components = data.shape[1]

    st.subheader("Explained Variance Ratio")
    st.write("""
    The plot below shows the percentage of variance explained by each of the principal components. 
             This helps to understand how much information each component captures from the data.
    """)
    pca_components = pca.explained_variance_ratio_[:num_components]
    fig, ax = plt.subplots()
    ax.bar(range(1, num_components + 1), pca_components, alpha=0.6, color='b')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Explained Variance by Each Principal Component')
    
    ax.set_xticks(range(1, num_components + 1))
    ax2 = ax.twinx() 
    ax2.plot(range(1, num_components + 1), np.cumsum(pca_components)*100, marker='o', c='red')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.grid(True)
    st.pyplot(fig)

def plot_data(data: pd.DataFrame):
    """Plot data with classes for first two dimensions
    Args:
        data (pd.DataFrame): _description_
    """
    # Scatter original data with first two columns
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)
    fig = px.scatter(data, x=data.iloc[:,0], y=data.iloc[:,1], color=data.columns[-1])
    st.plotly_chart(fig)