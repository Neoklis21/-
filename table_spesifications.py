import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Data Analysis Application')


st.sidebar.header('Upload your data')

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        
        st.write("Data successfully loaded!")
        
        
        st.dataframe(data)
        
        
        st.write("Shape of the data:", data.shape)
        st.write("Basic statistics:")
        st.write(data.describe())
        
        
        st.sidebar.subheader('Scatter Plot')
        features = data.columns[:-1]  # assuming the last column is the label
        x_axis = st.sidebar.selectbox('Feature for X-axis', features)
        y_axis = st.sidebar.selectbox('Feature for Y-axis', features)
        
        if x_axis and y_axis:
            st.subheader('Scatter Plot')
            fig, ax = plt.subplots()
            ax.scatter(data[x_axis], data[y_axis], alpha=0.7)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f'Scatter plot between {x_axis} and {y_axis}')
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Awaiting for file to be uploaded.")


