import streamlit as st
import pandas as pd


st.title('Data Analysis Application')


st.sidebar.header('Upload your data')

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
       
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        # Load Excel files
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        
        st.write("Data successfully loaded!")
        
    
        st.dataframe(data)
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Awaiting for file to be uploaded.")



