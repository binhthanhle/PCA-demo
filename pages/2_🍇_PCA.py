import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from src.plot_libs import plot_explain

def perform_pca(dataset):
    # Perform PCA
    options = st.slider("Select the number of component 👇", 1, dataset.shape[1]-1, 1)

    # Create a DataFrame with original features
    data = dataset.iloc[:,:-1]
 
    pca = PCA(n_components=options)
    pca_components = pca.fit_transform(data)

    # Create new DataFrame with PCA components
    list_pca = [f"pc{i}" for i in range(options)]
    pca_df = pd.DataFrame(pca_components, columns=list_pca)
    return pca, pca_df

def main():
    st.title('Performing PCA data')

    # Load dataset
    dataset = st.session_state.df
    
    pca, pca_df = perform_pca(dataset)
    st.subheader("Explain PCA")
    st.write('Principal Component Analysis (PCA) components:',
              pca.n_components_ , 
              ";\nTotal explained variance = ",
                round(pca.explained_variance_ratio_.sum(),5)  
            )
    st.dataframe(pca_df.head(3))
    fig, ax = plot_explain(pca)
    st.subheader("Plot the explain of PCA")
    st.pyplot(fig)

if __name__ == '__main__':
    main()