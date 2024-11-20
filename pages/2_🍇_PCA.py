import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from src.plot_libs import plot_explain



def perform_pca(dataset):
    # Perform PCA
    # Create a DataFrame with original features
    data = dataset.iloc[:,:-1]
 
    pca = PCA(n_components=dataset.shape[1]-1)
    pca_components = pca.fit_transform(data)

    # Create new DataFrame with PCA components
    list_pca = [f"pc{i}" for i in range(dataset.shape[1]-1)]
    pca_df = pd.DataFrame(pca_components, columns=list_pca)
    slider_num_components = st.sidebar.slider("Select the number of component 👇", 1, data.shape[1])
    plot_explain(data=data, pca=pca, num_components=slider_num_components)

    return pca, pca_df

def main():
    st.title('Performing PCA data')
    # Load dataset
    dataset = st.session_state.df
    dataset = dataset.reset_index(drop=True)
    
    pca, pca_df = perform_pca(dataset)
    pca_df = pd.concat([pca_df, dataset[dataset.columns[-1]]], axis=1)
    st.session_state.pca_df = pca_df

if __name__ == '__main__':
    main()