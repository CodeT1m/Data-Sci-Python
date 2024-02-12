import os
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import seaborn as sns

def main():
    st.title('Common ML Dataset Explorer')
    st.subheader('Simple Data Science Explorer with Streamlit')

    html_temp = """
    <div-style="background-color: tomato;"><p style="color:white; font-size:50px;">Streamlit is Awesome</p></div>

    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    def file_selector(folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select A file', filenames)
        return os.path.join(folder_path, selected_filename)
    
    filename = file_selector()
    st.info('You Seleted {}'.format(filename))
    
    #  Read Data
    df = pd.read_csv(filename)
    # Show Dataset
    if st.checkbox('Show Dataset'):
        number = st.number_input('Number of Rows to View',5,10)
        st.datafrane(df.head(number))
    # Show Columns
    if st.button('Column Name'):
        st.write(df.columns)
    
    # Show Shape
    if st.checkbox('Shape of Dataset'):
        st.write(df.shape)
        data_dim = st.radio('Show Dimension By ',('Rows', 'Columns'))
        if data_dim == 'Row':
            st.text('Number of Columns')
            st.write(df.shape[0])
        if data_dim == 'Columns':
            st.text('Number of Columns')
            st.write(df.shape[1])
        else:
            st.write(df.shape)
    # Show Values
    
    # Select Columns
    
    # Show Summary
    
if __name__ == '__main__':
    main()