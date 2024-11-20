import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.plot_libs import plot_data


def main():
    # Scatter original data with first two columns with classes
    st.title('Scatter Original Data')

    # Load dataset
    dataset = st.session_state.df
    pca_dataset = st.session_state.pca_df

    st.subheader("Plot first two dimensions of dataset")
    plot_data(dataset)

    st.write("___")
    st.subheader("Plot first two dimensions of PCA dataset")
    plot_data(pca_dataset)



if __name__ == '__main__':
    main()