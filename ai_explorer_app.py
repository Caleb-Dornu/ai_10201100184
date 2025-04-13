import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import PyPDF2

st.set_page_config(page_title="ML & AI Explorer", layout="wide")

st.sidebar.title("ML & AI Explorer")
task = st.sidebar.radio("Choose a Task", ["Regression", "Clustering", "Neural Network", "LLM Q&A"])

if task == "Regression":
    st.title("Regression Explorer")
    uploaded_file = st.file_uploader("Upload CSV for Regression", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())
        target_col = st.text_input("Enter the Target Column Name")
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            st.write(f"**Mean Absolute Error:** {mae:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")

            fig, ax = plt.subplots()
            ax.scatter(y, predictions)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs. Predicted")
            st.pyplot(fig)

            st.subheader("Make Custom Prediction")
            input_data = {}
            for col in X.columns:
                input_data[col] = st.number_input(f"{col}", value=0.0)
            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                pred = model.predict(input_df)[0]
                st.success(f"Predicted Value: {pred:.2f}")

elif task == "Clustering":
    st.title("Clustering Explorer")
    uploaded_file = st.file_uploader("Upload CSV for Clustering", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        cluster_cols = st.multiselect("Select Features for Clustering", df.columns.tolist())
        if cluster_cols:
            X = df[cluster_cols]
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters)
            df['Cluster'] = kmeans.fit_predict(X)

            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(X)
                X_plot = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
                X_plot['Cluster'] = df['Cluster']
            else:
                X_plot = X.copy()
                X_plot['Cluster'] = df['Cluster']

            fig, ax = plt.subplots()
            sns.scatterplot(data=X_plot, x=X_plot.columns[0], y=X_plot.columns[1], hue='Cluster', palette='tab10', ax=ax)
            st.pyplot(fig)

            st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")

elif task == "Neural Network":
    st.title("Neural Network Trainer")
    uploaded_file = st.file_uploader("Upload CSV for Classification", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        target_col = st.text_input("Enter the Target Column Name")
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = to_categorical(df[target_col].astype('category').cat.codes)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            lr = st.slider("Learning Rate", 0.001, 0.01, 0.005)
            epochs = st.slider("Epochs", 5, 50, 10)

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X.shape[1],)),
                Dense(32, activation='relu'),
                Dense(y.shape[1], activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=0)

            st.line_chart({"Training Accuracy": history.history['accuracy'], "Validation Accuracy": history.history['val_accuracy']})

            st.subheader("Upload Sample to Predict")
            sample = st.file_uploader("Sample CSV for Prediction", type="csv")
            if sample:
                sample_df = pd.read_csv(sample)
                pred = model.predict(sample_df)
                st.write("Predictions:", np.argmax(pred, axis=1))

else:
    st.title("LLM Question & Answer")
    st.info("This is a demo using a simple RAG-style LLM Q&A interface.")

    uploaded_pdf = st.file_uploader("Upload PDF for Q&A", type="pdf")
    if uploaded_pdf:
        pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
        full_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        st.success("PDF Processed. You can now ask questions.")

        question = st.text_input("Ask a question based on the document:")
        if question:
            # Placeholder for real LLM - simulated answer
            st.write("Answer:", f"This would be the answer to '{question}' based on the document content.")
            st.caption("Confidence Score: 0.92 (simulated)")
