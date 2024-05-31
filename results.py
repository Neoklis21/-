import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, silhouette_score, confusion_matrix, classification_report
import seaborn as sns


st.title('Data Analysis and Machine Learning Application')


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
       
     
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Scatter Plot", "2D Visualization", "EDA", "Classification", "Clustering", "Results and Comparison"])
       
        with tab1:
          
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

        with tab4:
            st.subheader('Classification Algorithms')
           
          
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

           
            st.write("K-Nearest Neighbors Classifier")
            k = st.slider('Select number of neighbors (k)', 1, 20, 5)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred_knn = knn.predict(X_test)
            accuracy_knn = accuracy_score(y_test, y_pred_knn)
            st.write(f"Accuracy: {accuracy_knn:.2f}")

           
            st.write("Confusion Matrix for KNN:")
            cm_knn = confusion_matrix(y_test, y_pred_knn)
            st.write(cm_knn)
            st.write("Classification Report for KNN:")
            cr_knn = classification_report(y_test, y_pred_knn)
            st.text(cr_knn)

         
            st.write("Decision Tree Classifier")
            max_depth = st.slider('Select max depth of the tree', 1, 20, 5)
            dt = DecisionTreeClassifier(max_depth=max_depth)
            dt.fit(X_train, y_train)
            y_pred_dt = dt.predict(X_test)
            accuracy_dt = accuracy_score(y_test, y_pred_dt)
            st.write(f"Accuracy: {accuracy_dt:.2f}")

            
            st.write("Confusion Matrix for Decision Tree:")
            cm_dt = confusion_matrix(y_test, y_pred_dt)
            st.write(cm_dt)
            st.write("Classification Report for Decision Tree:")
            cr_dt = classification_report(y_test, y_pred_dt)
            st.text(cr_dt)

        with tab5:
            st.subheader('Clustering Algorithms')

            st.write("K-Means Clustering")
            k_clusters = st.slider('Select number of clusters (k)', 2, 10, 3)
            kmeans = KMeans(n_clusters=k_clusters, random_state=42)
            clusters_kmeans = kmeans.fit_predict(X)
            silhouette_kmeans = silhouette_score(X, clusters_kmeans)
            st.write(f"Silhouette Score: {silhouette_kmeans:.2f}")

          
            st.write("Gaussian Mixture Model")
            n_components = st.slider('Select number of components', 2, 10, 3)
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            clusters_gmm = gmm.fit_predict(X)
            silhouette_gmm = silhouette_score(X, clusters_gmm)
            st.write(f"Silhouette Score: {silhouette_gmm:.2f}")

        with tab6:
            st.subheader('Results and Comparison')
           
         
            st.write("## Classification Results")
            st.write("### K-Nearest Neighbors")
            st.write(f"Accuracy: {accuracy_knn:.2f}")
            st.write("Confusion Matrix:")
            st.write(cm_knn)
            st.write("Classification Report:")
            st.text(cr_knn)
           
            st.write("### Decision Tree")
            st.write(f"Accuracy: {accuracy_dt:.2f}")
            st.write("Confusion Matrix:")
            st.write(cm_dt)
            st.write("Classification Report:")
            st.text(cr_dt)
           
            if accuracy_knn > accuracy_dt:
                st.write("**K-Nearest Neighbors performs better based on accuracy.**")
            else:
                st.write("**Decision Tree performs better based on accuracy.**")
           
            st.write("## Clustering Results")
            st.write("### K-Means")
            st.write(f"Silhouette Score: {silhouette_kmeans:.2f}")
           
            st.write("### Gaussian Mixture Model")
            st.write(f"Silhouette Score: {silhouette_gmm:.2f}")
           
            if silhouette_kmeans > silhouette_gmm:
                st.write("**K-Means performs better based on Silhouette Score.**")
            else:
                st.write("**Gaussian Mixture Model performs better based on Silhouette Score.**")
       
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Awaiting for file to be uploaded.")
