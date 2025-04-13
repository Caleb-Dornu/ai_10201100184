# ai_explorer_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
import tempfile
import plotly.graph_objects as go

# ==============================================
# Regression Section
# ==============================================
def regression_section():
    st.header("Regression Problem Solver")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your regression dataset (CSV)", type="csv", key="regression_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Target selection
        target_col = st.text_input("Enter the name of the target column", key="regression_target")
        
        if target_col and target_col in df.columns:
            # Feature selection
            features = st.multiselect("Select features for regression", 
                                    [col for col in df.columns if col != target_col],
                                    key="regression_features")
            
            if features:
                X = df[features]
                y = df[target_col]
                
                # Train-test split
                test_size = st.slider("Test set size (%)", 10, 40, 20, key="regression_test_size")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42)
                
                # Model training
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Evaluation
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                col1.metric("Mean Absolute Error", f"{mae:.2f}")
                col2.metric("R² Score", f"{r2:.2f}")
                
                # Visualization
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Actual vs Predicted')
                st.pyplot(fig)
                
                # Custom prediction
                st.subheader("Make Custom Predictions")
                input_data = {}
                for feature in features:
                    input_data[feature] = st.number_input(
                        f"Enter {feature}", 
                        value=float(X[feature].mean()),
                        key=f"regression_input_{feature}"
                    )
                
                if st.button("Predict", key="regression_predict"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)
                    st.success(f"Predicted {target_col}: {prediction[0]:.2f}")

# ==============================================
# Clustering Section
# ==============================================
def clustering_section():
    st.header("Data Clustering Explorer")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your clustering dataset (CSV)", type="csv", key="clustering_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Feature selection
        features = st.multiselect("Select features for clustering", 
                                df.columns,
                                key="clustering_features")
        
        if features:
            X = df[features]
            
            # Data scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Number of clusters
            n_clusters = st.slider("Select number of clusters", 2, 10, 3, key="n_clusters")
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            df['Cluster'] = clusters
            
            # Visualization
            st.subheader("Cluster Visualization")
            
            if len(features) >= 2:
                # 2D visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig = px.scatter(
                    x=X_pca[:, 0], y=X_pca[:, 1], 
                    color=clusters,
                    title="Cluster Visualization (PCA Reduced)",
                    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'}
                )
                st.plotly_chart(fig)
                
            if len(features) >= 3:
                # 3D visualization
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X_scaled)
                
                fig = px.scatter_3d(
                    x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                    color=clusters,
                    title="3D Cluster Visualization (PCA Reduced)",
                    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}
                )
                st.plotly_chart(fig)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            cluster_stats = df.groupby('Cluster')[features].mean()
            st.dataframe(cluster_stats)
            
            # Download clustered data
            st.download_button(
                label="Download Clustered Data",
                data=df.to_csv(index=False),
                file_name='clustered_data.csv',
                mime='text/csv',
                key="clustering_download"
            )

# ==============================================
# Neural Network Section
# ==============================================
def neural_net_section():
    st.header("Neural Network Classifier")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your classification dataset (CSV)", type="csv", key="nn_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Target selection
        target_col = st.text_input("Enter the name of the target column", key="nn_target")
        
        if target_col and target_col in df.columns:
            # Check if classification or regression
            problem_type = st.radio("Problem type", 
                                  ["Classification", "Regression"],
                                  key="problem_type")
            
            # Feature selection
            features = st.multiselect("Select features for the model", 
                                    [col for col in df.columns if col != target_col],
                                    key="nn_features")
            
            if features:
                X = df[features]
                y = df[target_col]
                
                # Preprocessing
                if problem_type == "Classification":
                    # Encode labels
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    y_processed = to_categorical(y_encoded)
                    n_classes = len(le.classes_)
                else:
                    y_processed = y.values.reshape(-1, 1)
                    n_classes = 1
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train-test split
                test_size = st.slider("Test set size (%)", 10, 40, 20, key="nn_test_size")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_processed, test_size=test_size/100, random_state=42)
                
                # Model configuration
                st.subheader("Model Configuration")
                n_layers = st.slider("Number of hidden layers", 1, 5, 2, key="n_layers")
                layer_sizes = []
                for i in range(n_layers):
                    size = st.slider(
                        f"Neurons in layer {i+1}", 
                        16, 256, 64, 
                        key=f"layer_{i}_size"
                    )
                    layer_sizes.append(size)
                
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, key="dropout_rate")
                learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, key="learning_rate")
                epochs = st.slider("Epochs", 10, 200, 50, key="epochs")
                batch_size = st.slider("Batch size", 16, 128, 32, key="batch_size")
                
                # Build model
                model = Sequential()
                model.add(Dense(layer_sizes[0], activation='relu', input_shape=(X_train.shape[1],)))
                
                for size in layer_sizes[1:]:
                    model.add(Dense(size, activation='relu'))
                    model.add(Dropout(dropout_rate))
                
                if problem_type == "Classification":
                    model.add(Dense(n_classes, activation='softmax'))
                    loss = 'categorical_crossentropy'
                    metrics = ['accuracy']
                else:
                    model.add(Dense(1))
                    loss = 'mse'
                    metrics = ['mae']
                
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
                # Model summary
                st.subheader("Model Architecture")
                with st.expander("Show model summary"):
                    model.summary(print_fn=lambda x: st.text(x))
                
                # Training
                if st.button("Train Model", key="train_model"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Callback for updating progress
                    class TrainingCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.4f}")
                    
                    # Early stopping
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[TrainingCallback(), early_stopping],
                        verbose=0
                    )
                    
                    # Training curves
                    st.subheader("Training Progress")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(history.history['loss'], label='Training Loss')
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_title('Loss Over Epochs')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    
                    if problem_type == "Classification":
                        ax2.plot(history.history['accuracy'], label='Training Accuracy')
                        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                        ax2.set_title('Accuracy Over Epochs')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Accuracy')
                    else:
                        ax2.plot(history.history['mae'], label='Training MAE')
                        ax2.plot(history.history['val_mae'], label='Validation MAE')
                        ax2.set_title('MAE Over Epochs')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('MAE')
                    
                    ax2.legend()
                    st.pyplot(fig)
                    
                    # Evaluation
                    st.subheader("Model Evaluation")
                    evaluation = model.evaluate(X_test, y_test, verbose=0)
                    
                    if problem_type == "Classification":
                        st.metric("Test Accuracy", f"{evaluation[1]*100:.2f}%")
                    else:
                        st.metric("Test MAE", f"{evaluation[1]:.4f}")
                    
                    # Prediction interface
                    st.subheader("Make Predictions")
                    input_data = {}
                    for feature in features:
                        input_data[feature] = st.number_input(
                            f"Enter {feature}", 
                            value=float(X[feature].mean()),
                            key=f"nn_input_{feature}"
                        )
                    
                    if st.button("Predict", key="nn_predict"):
                        input_array = np.array([[input_data[feature] for feature in features]])
                        input_scaled = scaler.transform(input_array)
                        prediction = model.predict(input_scaled)
                        
                        if problem_type == "Classification":
                            predicted_class = le.inverse_transform([np.argmax(prediction)])
                            st.success(f"Predicted class: {predicted_class[0]}")
                        else:
                            st.success(f"Predicted value: {prediction[0][0]:.4f}")

# ==============================================
# LLM Section (RAG Approach)
# ==============================================
def draw_rag_architecture():
    """Visualize the RAG architecture"""
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=[1, 2, 3, 4],
        y=[1, 1, 1, 1],
        mode="markers+text",
        marker=dict(size=30, color=["blue", "green", "red", "purple"]),
        text=["Document", "Vector DB", "LLM", "User"],
        textposition="top center"
    ))
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=[1, 2], y=[1, 1],
        mode="lines",
        line=dict(width=2, color="gray")
    ))
    fig.add_trace(go.Scatter(
        x=[2, 3], y=[1, 1],
        mode="lines",
        line=dict(width=2, color="gray")
    ))
    fig.add_trace(go.Scatter(
        x=[3, 4], y=[1, 1],
        mode="lines",
        line=dict(width=2, color="gray")
    ))
    
    # Add annotations
    fig.add_annotation(x=1.5, y=1.1, text="Chunk & Embed", showarrow=False)
    fig.add_annotation(x=2.5, y=1.1, text="Retrieve", showarrow=False)
    fig.add_annotation(x=3.5, y=1.1, text="Generate", showarrow=False)
    
    fig.update_layout(
        title="RAG Architecture",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def llm_section():
    st.header("Large Language Model (RAG) Q&A System")
    
    # Architecture visualization
    st.subheader("System Architecture")
    st.plotly_chart(draw_rag_architecture())
    
    # Methodology explanation
    with st.expander("Methodology Details"):
        st.markdown("""
        ### Retrieval-Augmented Generation (RAG) Approach
        
        1. **Document Processing**:
           - Uploaded PDF documents are parsed and split into manageable chunks
           - Each chunk is embedded into a vector space using HuggingFace embeddings
        
        2. **Vector Database**:
           - Chunk embeddings are stored in a FAISS vector store for efficient similarity search
           - Enables quick retrieval of relevant document passages for any query
        
        3. **LLM Integration**:
           - Mistral-7B model is used for answer generation
           - Retrieved document chunks provide context to the LLM
           - The model synthesizes information from context to answer user queries
        
        4. **Advantages**:
           - More accurate than pure LLM responses as it's grounded in documents
           - Can handle domain-specific questions beyond the LLM's training data
           - More transparent as answers can be traced back to source documents
        """)
    
    # Document upload
    st.subheader("Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF document (e.g., Academic City Student Policy)", 
                                   type="pdf", 
                                   key="llm_upload")
    
    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Process PDF
        pdf_reader = PdfReader(tmp_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Initialize LLM
        if 'HUGGINGFACEHUB_API_TOKEN' in st.secrets:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        else:
            st.warning("Please set HUGGINGFACEHUB_API_TOKEN in Streamlit secrets to use the LLM")
            return
        
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.5, "max_length": 1024}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=knowledge_base.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Q&A interface
        st.subheader("Ask Questions")
        question = st.text_input("Enter your question about the document", key="llm_question")
        
        if question:
            with st.spinner("Searching for answers..."):
                result = qa_chain({"query": question})
                
            st.subheader("Answer")
            st.write(result["result"])
            
            # Show source documents
            with st.expander("See source documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")

# ==============================================
# Main Application
# ==============================================
def main():
    st.set_page_config(page_title="AI Explorer", layout="wide")
    
    st.sidebar.title("AI Explorer")
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("Select Task", [
        "Regression",
        "Clustering",
        "Neural Network",
        "LLM Q&A"
    ])
    
    st.title("AI/ML Problem Explorer")
    
    if app_mode == "Regression":
        regression_section()
    elif app_mode == "Clustering":
        clustering_section()
    elif app_mode == "Neural Network":
        neural_net_section()
    elif app_mode == "LLM Q&A":
        llm_section()

if __name__ == "__main__":
    main()