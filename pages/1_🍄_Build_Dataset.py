import streamlit as st

from data.build_data import build_dataset, build_iris_data
from src.plot_libs import hist_plot



def main():
    st.header('Build dataset with number of classes')
    # Load dataset
    options_dataset = st.radio("", ['Iris Data', 'Manual Data'], horizontal=True)
    st.write("------------------------------------------------")
    if options_dataset == 'Iris Data':
        dataset = build_iris_data()
    else:
        slider_no_class = st.sidebar.number_input('Choose number of classes', min_value=1, max_value=3)
        slider_no_feature = st.sidebar.number_input('Choose number of feature', min_value=1, max_value=10)
        dataset = build_dataset(no_class=slider_no_class, no_features=slider_no_feature)
        
    st.subheader(f'Data {options_dataset} Summary')
    # Create the histograms plots
    hist_plot(dataset=dataset)

    # Store dataset in session state
    st.session_state.df = dataset

if __name__ == '__main__':
    main()
