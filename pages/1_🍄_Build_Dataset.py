import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

from data.build_data import build_dataset
from src.plot_libs import plot_hist

def main():
    st.header('Build dataset with number of classes')
    # Load dataset
    options = st.slider('Choose number of classes', 1, 5, 1)
    dataset = build_dataset(no_class=options)

    st.subheader('Data Summary')
    st.subheader('Plotting data distribution')

    # Create the histograms plots
    feature = st.sidebar.selectbox("Choose a feature", 
                        dataset.columns.to_list())
    fig = plot_hist(dataset,feature_name=feature, class_labels=dataset.columns[-1])
    st.plotly_chart(fig, use_container_width=True)

    # Store dataset in session state
    st.session_state.df = dataset

if __name__ == '__main__':
    main()
