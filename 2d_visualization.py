import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


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
        
        
        tab1, tab2, tab3 = st.tabs(["Scatter Plot", "2D Visualization", "EDA"])
        
        with tab1:
           
            st.sidebar.subheader('Scatter Plot')
            features = data.columns[:-1]  
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
        
        with tab2:
            st.subheader('2D Visualization using Dimensionality Reduction')
            algo = st.selectbox("Choose a dimensionality reduction algorithm", ["PCA", "t-SNE"])
            if st.button("Generate 2D Plot"):
                if algo == "PCA":
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(data.iloc[:, :-1])
                elif algo == "t-SNE":
                    tsne = TSNE(n_components=2)
                    components = tsne.fit_transform(data.iloc[:, :-1])
                
                fig, ax = plt.subplots()
                scatter = ax.scatter(components[:, 0], components[:, 1], c=data.iloc[:, -1], cmap='viridis', alpha=0.7)
                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend1)
                ax.set_title(f'2D Visualization using {algo}')
                st.pyplot(fig)
        
        with tab3:
            st.subheader('Exploratory Data Analysis (EDA)')
            
            
            st.write("Distribution of the label column:")
            fig, ax = plt.subplots()
            sns.countplot(x=data.iloc[:, -1], ax=ax)
            ax.set_title('Distribution of Labels')
            st.pyplot(fig)
            
           
            st.write("Pairplot of features:")
            pairplot_fig = sns.pairplot(data, hue=data.columns[-1])
            st.pyplot(pairplot_fig)
            
            
            st.write("Correlation heatmap:")
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Awaiting for file to be uploaded.")

