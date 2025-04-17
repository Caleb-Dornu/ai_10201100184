import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import Callback
import os
import time
from dotenv import load_dotenv

# --- LLM RAG Imports ---
# Use try-except blocks for optional dependencies if needed
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings # Use new integration
    # from langchain_community.embeddings import HuggingFaceEmbeddings # Older path
    from langchain_huggingface import HuggingFaceEndpoint # New integration for HF API LLMs
    # from langchain_community.llms import HuggingFaceHub # Older path
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError as e:
    st.error(f"Please install required LangChain packages: {e}")
    st.stop()

# --- Load Environment Variables (for Hugging Face API Key) ---
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Page Configuration ---
st.set_page_config(
    page_title="ML & AI Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
@st.cache_data # Cache data loading
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

@st.cache_resource # Cache model training or heavy resources
def train_regression_model(df, target_column):
    try:
        X = df.drop(target_column, axis=1)
        # Select only numeric features for simplicity in this example
        X = X.select_dtypes(include=np.number)
        if X.empty:
            st.error("No numeric features found for regression after dropping the target column.")
            return None, None, None, None, None
        y = df[target_column]

        # Handle missing values (simple mean imputation)
        X = X.fillna(X.mean())
        y = y.fillna(y.mean()) # Also impute target just in case

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, X_test, y_test, y_pred, mae, r2, X.columns.tolist() # Return feature names used
    except KeyError:
        st.error(f"Error: Target column '{target_column}' not found in the uploaded file.")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None, None, None, None, None, None

@st.cache_resource
def perform_clustering(df, n_clusters, selected_features):
    try:
        if not selected_features:
            st.warning("Please select at least one feature for clustering.")
            return None, None, None

        X = df[selected_features].copy() # Work on a copy
        # Handle missing values (simple mean imputation)
        X = X.fillna(X.mean())

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init explicitly
        kmeans.fit(X_scaled)

        cluster_labels = kmeans.labels_
        centroids_scaled = kmeans.cluster_centers_

        # Inverse transform centroids to original scale for interpretation (optional but good)
        # centroids = scaler.inverse_transform(centroids_scaled)

        return cluster_labels, centroids_scaled, scaler # Return scaler for potential later use

    except Exception as e:
        st.error(f"An error occurred during clustering: {e}")
        return None, None, None

# Custom Keras Callback for Streamlit progress
class StreamlitCallback(Callback):
    def __init__(self):
        self.progress_bar = st.progress(0)
        self.epoch_text = st.empty()
        self.metrics_placeholder = st.empty() # Placeholder for loss/accuracy charts

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_progress = (epoch + 1) / self.params['epochs']
        self.progress_bar.progress(min(float(current_progress), 1.0)) # Ensure progress doesn't exceed 1.0
        epoch_log = f"Epoch {epoch+1}/{self.params['epochs']}"
        metrics_log = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.epoch_text.text(f"{epoch_log} | {metrics_log}")

        # Store history (crude way for live update, better to plot after training)
        if not hasattr(self, 'history'):
            self.history = {k: [] for k in logs.keys()}
        for k, v in logs.items():
            self.history[k].append(v)

        # Update plot data (if plotting live) - simplified: plot after training
        # self._plot_metrics() # Call a plotting function if doing live plots


    def on_train_end(self, logs=None):
        self.progress_bar.empty() # Remove progress bar on completion
        self.epoch_text.empty() # Clear epoch text

    # Example plotting function (call this after training is done)
    # def _plot_metrics(self):
    #    with self.metrics_placeholder.container():
    #        # Plot logic using self.history
    #        pass

@st.cache_resource(show_spinner="Training Neural Network...")
def train_nn_model(df, target_column, epochs=10, learning_rate=0.001):
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Preprocessing
        # Select numeric features and scale them
        numeric_features = X.select_dtypes(include=np.number).columns
        if len(numeric_features) > 0:
            scaler = StandardScaler()
            X[numeric_features] = scaler.fit_transform(X[numeric_features])

        # Handle categorical features (simple one-hot encoding)
        categorical_features = X.select_dtypes(include='object').columns
        if len(categorical_features) > 0:
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True) # Use pandas for simplicity

        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        n_classes = len(label_encoder.classes_)
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=n_classes)

        # Handle missing values after potential encoding/scaling
        X = X.fillna(0) # Simple fillna with 0 after potential scaling/encoding

        # Ensure all data is numeric for TF
        X = X.astype(np.float32)

        X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

        # Build Model
        model = Sequential([
            Input(shape=(X_train.shape[1],)), # Input layer based on number of features
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(n_classes, activation='softmax') # Softmax for multi-class classification
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # --- Training with Callback Placeholder ---
        st_callback = StreamlitCallback()
        # Note: The callback updates progress but plotting is done post-training here
        status_placeholder = st.empty()
        status_placeholder.info("Training neural network...")
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            validation_data=(X_val, y_val),
                            callbacks=[st_callback],
                            batch_size=32,
                            verbose=0) # Use verbose=0 as StreamlitCallback handles output
        status_placeholder.success("Training complete!")


        return model, history.history, scaler, label_encoder, X.columns.tolist(), numeric_features.tolist(), categorical_features.tolist()

    except KeyError:
        st.error(f"Error: Target column '{target_column}' not found.")
        return None, None, None, None, None, None, None
    except ValueError as ve:
        st.error(f"Data Preprocessing Error: {ve}. Check if your target variable needs encoding or if features have non-numeric data.")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during NN training: {e}")
        return None, None, None, None, None, None, None

# --- LLM RAG Functions ---

# Use session state to store heavy objects like the QA chain
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'rag_setup_done' not in st.session_state:
    st.session_state.rag_setup_done = False

@st.cache_resource(show_spinner="Setting up RAG pipeline (takes a few moments)...")
def setup_rag_pipeline(pdf_path, repo_id="mistralai/Mistral-7B-Instruct-v0.1", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Sets up the LangChain RAG pipeline."""
    if not HUGGINGFACEHUB_API_TOKEN:
        st.error("Hugging Face API Token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
        st.stop()

    try:
        # 1. Load Data
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            st.error("Could not load any documents from the PDF.")
            return None

        # 2. Split Data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        if not texts:
            st.error("Could not split documents into chunks.")
            return None

        # 3. Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # 4. Vector Store
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
        except ImportError:
            st.error("FAISS library not found or properly installed. Please install 'faiss-cpu' or 'faiss-gpu'.")
            return None

        # 5. LLM
        # Use HuggingFaceEndpoint for API access
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_new_tokens=512, # Adjust as needed
            temperature=0.1, # Lower temperature for more factual Q&A
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
        )
        # Older HuggingFaceHub method (might require different parameters)
        # llm = HuggingFaceHub(repo_id=repo_id,
        #                      model_kwargs={"temperature": 0.1, "max_length": 512},
        #                      huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)


        # 6. Retrieval Chain with Custom Prompt
        template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
        Be concise and answer based *only* on the provided text.

        Context:
        {context}

        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True # Option to return sources
        )

        return qa_chain

    except Exception as e:
        st.error(f"Error setting up RAG pipeline: {e}")
        return None

# --- Streamlit App Layout ---

st.sidebar.title("🧭 ML & AI Explorer")
app_mode = st.sidebar.selectbox(
    "Choose Task",
    ["Welcome", "Regression", "Clustering", "Neural Network (Classification)", "LLM Q&A (RAG)"]
)

# --- Welcome Page ---
if app_mode == "Welcome":
    st.title("Welcome to the ML & AI Explorer!")
    st.markdown("""
        This application allows you to explore different machine learning and AI tasks interactively.
        Use the sidebar to navigate between tasks:

        * **Regression:** Predict continuous values (e.g., house prices).
        * **Clustering:** Group similar data points together (e.g., customer segmentation).
        * **Neural Network:** Perform classification tasks using a simple Feedforward Neural Network.
        * **LLM Q&A (RAG):** Ask questions about a specific document using a Large Language Model enhanced with Retrieval-Augmented Generation.

        Upload your data (CSV for Regression, Clustering, NN; PDF for LLM Q&A) and explore the results!
    """)
    st.info("Ensure your datasets are cleaned and properly formatted for the best results.")

# --- Regression Section ---
elif app_mode == "Regression":
    st.header("📈 Regression Analysis")
    st.markdown("Predict a continuous target variable based on input features.")

    uploaded_file = st.file_uploader("Upload your CSV data for Regression", type=["csv"])
    target_column_reg = st.text_input("Enter the exact name of the target column", key="reg_target", help="e.g., 'Price'")

    if uploaded_file is not None and target_column_reg:
        df_reg = load_data(uploaded_file)
        if df_reg is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df_reg.head())

            if target_column_reg in df_reg.columns:
                st.success(f"Target column '{target_column_reg}' found.")

                # Check if target is numeric
                if not pd.api.types.is_numeric_dtype(df_reg[target_column_reg]):
                     st.error(f"Target column '{target_column_reg}' is not numeric. Regression requires a numeric target.")
                else:
                    # Train Model
                    model_reg, X_test_reg, y_test_reg, y_pred_reg, mae_reg, r2_reg, feature_names_reg = train_regression_model(df_reg, target_column_reg)

                    if model_reg is not None:
                        st.subheader("Model Performance")
                        col1, col2 = st.columns(2)
                        col1.metric("Mean Absolute Error (MAE)", f"{mae_reg:.4f}")
                        col2.metric("R² Score", f"{r2_reg:.4f}")

                        st.subheader("Predictions vs. Actual Values")
                        fig_scatter = px.scatter(x=y_test_reg, y=y_pred_reg, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title="Actual vs. Predicted")
                        fig_scatter.add_trace(go.Scatter(x=[y_test_reg.min(), y_test_reg.max()], y=[y_test_reg.min(), y_test_reg.max()], mode='lines', name='Ideal Fit', line=dict(dash='dash')))
                        st.plotly_chart(fig_scatter, use_container_width=True)

                        # Basic feature importance (coefficients for Linear Regression)
                        st.subheader("Feature Importance (Model Coefficients)")
                        try:
                            coeffs = pd.DataFrame(model_reg.coef_, index=feature_names_reg, columns=['Coefficient'])
                            st.dataframe(coeffs.sort_values('Coefficient', ascending=False))
                        except Exception as e:
                            st.warning(f"Could not display coefficients: {e}")


                        st.subheader("Make a Prediction on Custom Data")
                        custom_input_reg = {}
                        st.markdown(f"Enter values for the features used in training: `{', '.join(feature_names_reg)}`")
                        cols_input = st.columns(len(feature_names_reg))
                        valid_input = True
                        for i, feature in enumerate(feature_names_reg):
                           with cols_input[i]:
                               # Use number_input for numeric features
                               try:
                                   custom_input_reg[feature] = st.number_input(f"{feature}", key=f"reg_input_{feature}", value=float(df_reg[feature].mean())) # Default to mean
                               except ValueError:
                                   st.error(f"Invalid input for {feature}. Please enter a number.")
                                   valid_input = False


                        if st.button("Predict House Price", key="predict_reg") and valid_input:
                            input_df = pd.DataFrame([custom_input_reg])
                            # Ensure columns are in the same order as training
                            input_df = input_df[feature_names_reg]
                            try:
                                prediction = model_reg.predict(input_df)
                                st.success(f"Predicted {target_column_reg}: {prediction[0]:.2f}")
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")

            else:
                st.error(f"Target column '{target_column_reg}' not found in the uploaded file. Available columns: {df_reg.columns.tolist()}")

# --- Clustering Section ---
elif app_mode == "Clustering":
    st.header("🧩 Clustering Analysis (K-Means)")
    st.markdown("Group data points into clusters based on feature similarity.")

    uploaded_file_clus = st.file_uploader("Upload your CSV data for Clustering", type=["csv"], key="cluster_upload")

    if uploaded_file_clus is not None:
        df_clus = load_data(uploaded_file_clus)
        if df_clus is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df_clus.head())

            # Allow user to select features for clustering
            numeric_cols = df_clus.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found in the dataset for clustering.")
            else:
                selected_features_clus = st.multiselect(
                    "Select numeric features for clustering",
                    options=numeric_cols,
                    default=numeric_cols[:min(len(numeric_cols), 5)] # Default to first 5 or fewer
                )

                if selected_features_clus:
                    n_clusters = st.slider("Select the number of clusters (K)", min_value=2, max_value=15, value=3, key="k_slider")

                    if st.button("Run K-Means Clustering", key="run_cluster"):
                        cluster_labels, centroids_scaled, scaler = perform_clustering(df_clus, n_clusters, selected_features_clus)

                        if cluster_labels is not None:
                            df_clus['Cluster'] = cluster_labels
                            st.success(f"Clustering complete! Data points assigned to {n_clusters} clusters.")

                            st.subheader("Clustering Results Visualization")
                            if len(selected_features_clus) == 2:
                                # 2D Scatter Plot
                                fig_clus_2d = px.scatter(
                                    df_clus,
                                    x=selected_features_clus[0],
                                    y=selected_features_clus[1],
                                    color='Cluster',
                                    color_continuous_scale=px.colors.qualitative.Vivid,
                                    title=f'K-Means Clustering (K={n_clusters}) - Features: {selected_features_clus[0]} vs {selected_features_clus[1]}'
                                )
                                # Add centroids (need to inverse transform them if plotting on original scale)
                                # centroids_orig = scaler.inverse_transform(centroids_scaled)
                                # fig_clus_2d.add_trace(go.Scatter(x=centroids_orig[:, 0], y=centroids_orig[:, 1], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Centroids'))
                                st.plotly_chart(fig_clus_2d, use_container_width=True)
                                st.caption("Centroids are based on scaled data and not shown directly on this plot of original data.")

                            elif len(selected_features_clus) > 2:
                                st.info("Dataset has more than 2 features. Performing PCA for 2D visualization.")
                                try:
                                    X_clus = df_clus[selected_features_clus].copy().fillna(df_clus[selected_features_clus].mean())
                                    X_clus_scaled = scaler.transform(X_clus) # Use the scaler from clustering
                                    pca = PCA(n_components=2)
                                    principal_components = pca.fit_transform(X_clus_scaled)
                                    pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
                                    pca_df['Cluster'] = cluster_labels

                                    fig_clus_pca = px.scatter(
                                        pca_df,
                                        x='Principal Component 1',
                                        y='Principal Component 2',
                                        color='Cluster',
                                        color_continuous_scale=px.colors.qualitative.Vivid,
                                        title=f'K-Means Clustering (K={n_clusters}) - PCA Visualization'
                                    )
                                    st.plotly_chart(fig_clus_pca, use_container_width=True)
                                except Exception as pca_e:
                                    st.error(f"Could not perform PCA for visualization: {pca_e}")
                            else:
                                st.warning("Select at least 2 features for visualization.")


                            st.subheader("Download Clustered Data")
                            csv_data = df_clus.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download data as CSV",
                                data=csv_data,
                                file_name=f'clustered_data_{n_clusters}_clusters.csv',
                                mime='text/csv',
                                key="download_cluster"
                            )
                else:
                    st.warning("Please select features to use for clustering.")

# --- Neural Network Section ---
elif app_mode == "Neural Network (Classification)":
    st.header("🧠 Neural Network for Classification")
    st.markdown("Train a simple Feedforward Neural Network for classification tasks.")

    uploaded_file_nn = st.file_uploader("Upload your CSV data for Classification", type=["csv"], key="nn_upload")
    target_column_nn = st.text_input("Enter the exact name of the target column", key="nn_target", help="e.g., 'Species' or 'Digit'")

    if uploaded_file_nn is not None and target_column_nn:
        df_nn = load_data(uploaded_file_nn)
        if df_nn is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df_nn.head())

            if target_column_nn in df_nn.columns:
                st.success(f"Target column '{target_column_nn}' found.")

                # Hyperparameters
                st.sidebar.subheader("NN Hyperparameters")
                epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=1000, value=20, step=1, key="nn_epochs")
                learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=0.001, step=1e-4, format="%.4f", key="nn_lr")

                if st.button("Train Neural Network", key="train_nn"):
                    model_nn, history_nn, scaler_nn, encoder_nn, feature_names_nn, numeric_features_nn, categorical_features_nn = train_nn_model(df_nn, target_column_nn, epochs, learning_rate)

                    if model_nn is not None and history_nn is not None:
                        st.subheader("Training History")

                        # Create DataFrame for plotting
                        history_df = pd.DataFrame(history_nn)
                        history_df['epoch'] = range(1, len(history_df) + 1)

                        # Plot Loss
                        fig_loss = px.line(history_df, x='epoch', y=['loss', 'val_loss'], title='Model Loss')
                        fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='Loss', legend_title='Metric')
                        st.plotly_chart(fig_loss, use_container_width=True)

                        # Plot Accuracy
                        fig_acc = px.line(history_df, x='epoch', y=['accuracy', 'val_accuracy'], title='Model Accuracy')
                        fig_acc.update_layout(xaxis_title='Epoch', yaxis_title='Accuracy', legend_title='Metric')
                        st.plotly_chart(fig_acc, use_container_width=True)

                        st.subheader("Make Predictions on New Data")
                        st.markdown(f"Provide input features ({len(feature_names_nn)} features used in training).")

                        # Allow file upload for batch prediction
                        test_file_nn = st.file_uploader("Upload a CSV file with test samples (same columns except target)", type=["csv"], key="nn_test_upload")
                        if test_file_nn:
                             df_test_nn = load_data(test_file_nn)
                             if df_test_nn is not None:
                                 st.write("Test Data Preview:")
                                 st.dataframe(df_test_nn.head())
                                 try:
                                     # Preprocess test data similarly to training data
                                     X_test_nn = df_test_nn.copy()
                                     if len(numeric_features_nn) > 0:
                                         X_test_nn[numeric_features_nn] = scaler_nn.transform(X_test_nn[numeric_features_nn])
                                     if len(categorical_features_nn) > 0:
                                         X_test_nn = pd.get_dummies(X_test_nn, columns=categorical_features_nn, drop_first=True)

                                     # Align columns with training data (handle missing/extra columns)
                                     X_test_nn = X_test_nn.reindex(columns=feature_names_nn, fill_value=0)
                                     X_test_nn = X_test_nn.fillna(0) # Fill any remaining NaNs
                                     X_test_nn = X_test_nn.astype(np.float32)

                                     # Predict
                                     predictions_proba = model_nn.predict(X_test_nn)
                                     predictions_indices = np.argmax(predictions_proba, axis=1)
                                     predictions_labels = encoder_nn.inverse_transform(predictions_indices)

                                     # Display results
                                     results_df = pd.DataFrame({
                                         'Prediction': predictions_labels,
                                         'Confidence': np.max(predictions_proba, axis=1)
                                     })
                                     st.write("Predictions:")
                                     st.dataframe(results_df)

                                 except Exception as e:
                                     st.error(f"Error during prediction on uploaded file: {e}")
                        else:
                            st.info("Or enter individual sample data below (not yet implemented for simplicity).")
                            # Implementation for single sample input would require creating input fields
                            # similar to the regression section, applying the scaler and one-hot encoding,
                            # aligning columns, and then predicting. This can be complex to manage dynamically.

            else:
                st.error(f"Target column '{target_column_nn}' not found. Available columns: {df_nn.columns.tolist()}")

# --- LLM Q&A (RAG) Section ---
elif app_mode == "LLM Q&A (RAG)":
    st.header("💬 LLM Question Answering (RAG)")
    st.markdown("Ask questions about a document using a Large Language Model combined with Retrieval-Augmented Generation.")
    st.markdown("---")
    st.subheader("1. Setup")

    # Using a fixed PDF path for this example. Replace with a file uploader if desired.
    # pdf_path_llm = "YOUR_PDF_FILE.pdf" # Replace with path to your PDF
    # Hardcoding the MOFEP link for demonstration, assuming it needs downloading first
    # In a real app, you might download this programmatically or ask user to upload.
    pdf_url_llm = "https://mofep.gov.gh/sites/default/files/budget-statements/2024-Budget-Statement-and-Economic-Policy_v4.pdf" # Using 2024 as 2025 might not exist yet
    pdf_filename = "2024_budget_ghana.pdf"

    st.info(f"Using document: **Ghana 2024 Budget Statement** (`{pdf_url_llm}`)")
    st.warning("Ensure you have set the `HUGGINGFACEHUB_API_TOKEN` in your environment variables or a `.env` file.")

    # Download the file if it doesn't exist (simple implementation)
    if not os.path.exists(pdf_filename):
        try:
            import requests
            st.info(f"Downloading {pdf_filename}...")
            response = requests.get(pdf_url_llm, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(pdf_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"Downloaded {pdf_filename}.")
        except Exception as e:
            st.error(f"Failed to download PDF: {e}")
            st.stop() # Stop if download fails

    # Button to trigger RAG setup
    if st.button("Initialize RAG Pipeline", key="init_rag"):
        if os.path.exists(pdf_filename):
            if HUGGINGFACEHUB_API_TOKEN:
                st.session_state.qa_chain = setup_rag_pipeline(pdf_filename)
                if st.session_state.qa_chain:
                    st.session_state.rag_setup_done = True
                    st.success("RAG Pipeline Initialized Successfully!")
                else:
                    st.error("Failed to initialize RAG Pipeline. Check logs.")
            else:
                 st.error("Hugging Face API Token not found. Cannot initialize RAG.")
        else:
            st.error(f"PDF file '{pdf_filename}' not found. Please ensure it's downloaded or available.")

    st.markdown("---")
    st.subheader("2. Ask a Question")

    if st.session_state.rag_setup_done and st.session_state.qa_chain:
        user_question = st.text_input("Enter your question about the document:", key="llm_question")

        if st.button("Get Answer", key="ask_llm"):
            if user_question:
                with st.spinner("Retrieving answer..."):
                    try:
                        start_time = time.time()
                        # Use invoke for the newer Langchain expression language standard
                        result = st.session_state.qa_chain.invoke({"query": user_question})
                        # Older chain call: result = st.session_state.qa_chain({"query": user_question})
                        end_time = time.time()

                        st.info(f"Answer (took {end_time - start_time:.2f} seconds):")
                        st.markdown(result['result']) # Accessing the answer field

                        # Display source documents (optional)
                        with st.expander("Show Relevant Source Chunks"):
                             if 'source_documents' in result:
                                 for i, doc in enumerate(result['source_documents']):
                                    st.write(f"**Chunk {i+1} (Page: {doc.metadata.get('page', 'N/A')})**")
                                    st.caption(doc.page_content[:500] + "...") # Show snippet
                             else:
                                 st.write("Source documents not available in the result.")

                    except Exception as e:
                        st.error(f"Error getting answer from LLM: {e}")
            else:
                st.warning("Please enter a question.")
    elif not st.session_state.rag_setup_done:
        st.warning("Please initialize the RAG pipeline first using the button above.")
    else:
         st.error("RAG pipeline is not available. Initialization might have failed.")

    st.markdown("---")
    st.subheader("3. RAG System Details")
    with st.expander("Show RAG Architecture and Methodology"):
        st.markdown("**LLM Approach:** Retrieval-Augmented Generation (RAG)")
        st.markdown("**Dataset:** Ghana 2024 Budget Statement and Economic Policy (PDF)")
        st.markdown("**LLM Model:** `mistralai/Mistral-7B-Instruct-v0.1` (via Hugging Face API)")
        st.markdown("**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`")
        st.markdown("**Vector Store:** FAISS (Facebook AI Similarity Search)")

        st.markdown("#### Architecture Diagram")
        # You would save your diagram as 'rag_architecture.png' in the 'assets' folder
        if os.path.exists("assets/rag_architecture.png"):
            st.image("assets/rag_architecture.png", caption="RAG Pipeline Architecture")
        else:
            st.warning("Architecture diagram 'assets/rag_architecture.png' not found.")
            st.markdown("""
            * **User Input:** Question is asked.
            * **Embedding:** Question is converted to a vector using `all-MiniLM-L6-v2`.
            * **Vector Store (FAISS):** Pre-computed embeddings of document chunks (from the Budget PDF) are stored. A similarity search finds chunks most relevant to the question's vector.
            * **Retrieval:** Top relevant text chunks are retrieved from the document.
            * **Prompt Augmentation:** The original question and the retrieved text chunks are combined into a detailed prompt for the LLM.
            * **LLM (Mistral-7B Instruct):** The LLM receives the augmented prompt and generates an answer based *primarily* on the provided context chunks.
            * **Output:** The generated answer is presented to the user.
            """)

        st.markdown("#### Methodology")
        st.markdown("""
        1.  **Data Loading:** The Ghana 2024 Budget PDF (`.pdf`) is loaded using `PyPDFLoader`. Each page is treated as a separate document initially.
        2.  **Text Splitting:** The text from the loaded documents is split into smaller, overlapping chunks (e.g., 1000 characters with 150 overlap) using `RecursiveCharacterTextSplitter`. This ensures semantic context is preserved across chunks and fits within the model's context limits.
        3.  **Embedding Generation:** Each text chunk is converted into a dense numerical vector (embedding) using the `sentence-transformers/all-MiniLM-L6-v2` model via `HuggingFaceEmbeddings`. These embeddings capture the semantic meaning of the text.
        4.  **Vector Storage (Indexing):** The text chunks and their corresponding embeddings are stored in a FAISS vector store. FAISS allows for efficient similarity searches based on vector distance.
        5.  **User Query & Retrieval:** When a user asks a question, it's also embedded using the same embedding model. This query embedding is used to search the FAISS index for the embeddings of the document chunks that are most similar (e.g., using cosine similarity or Euclidean distance). The corresponding text chunks are retrieved.
        6.  **Prompt Engineering:** A specific prompt template is used. It instructs the LLM (`Mistral-7B-Instruct-v0.1`) to answer the user's question *based only on the retrieved text chunks* (provided as context in the prompt). It's guided to state if the answer isn't found in the context rather than hallucinating.
        7.  **LLM Interaction:** The combined prompt (instructions + retrieved context + user question) is sent to the Mistral-7B model via the `HuggingFaceEndpoint`.
        8.  **Answer Generation:** The LLM processes the input and generates an answer, grounding its response in the provided budget document context. The `RetrievalQA` chain orchestrates this process.
        9.  **Response Display:** The final answer generated by the LLM is displayed to the user. Optionally, the source chunks used to generate the answer can also be shown for verification.
        """)

        st.markdown("#### Evaluation & Comparison with ChatGPT (Example)")
        st.markdown("""
        *Note: This is a qualitative comparison based on hypothetical runs. Actual results may vary.*

        **Sample Questions & Comparison:**

        1.  **Q:** What is the projected GDP growth rate for 2024 according to the budget statement?
            * **RAG System:** "Based on the provided context from the 2024 Budget Statement, the projected real GDP growth rate for 2024 is X.X%..." (Finds the specific figure if present in the text).
            * **ChatGPT (GPT-4/Web):** "According to the information available up to my last update, Ghana's projected GDP growth for 2024 was often cited around X-Y%. However, for the definitive figure from the *specific* 2024 Budget Statement, you should consult the document directly." (May give a general range or refer to the document, less likely to have the exact indexed PDF content).
            * **Analysis:** RAG is highly accurate *if* the information is explicitly in the document. ChatGPT provides broader context but might not have the specific, granular detail from this *exact* PDF unless it was part of its training data or accessed via Browse.

        2.  **Q:** What are the key priorities for the education sector mentioned in the budget?
            * **RAG System:** "The document outlines several priorities for education, including [lists specific points found in retrieved chunks, e.g., STEM focus, TVET expansion, infrastructure projects mentioned]."
            * **ChatGPT (GPT-4/Web):** "Ghana's typical education priorities often include improving access, quality, STEM education, and TVET. The 2024 budget likely detailed specific allocations and initiatives like [might list plausible but potentially generic or slightly outdated points]."
            * **Analysis:** RAG provides document-specific priorities. ChatGPT might generalize or synthesize information from various sources, potentially missing nuances or specific naming conventions used *only* in the 2024 budget.

        3.  **Q:** What is the government's stance on cryptocurrency regulation in this budget?
            * **RAG System:** "Based on the provided context, there is no mention of cryptocurrency regulation in the retrieved sections of the 2024 Budget Statement." (Correctly states if the info isn't found).
            * **ChatGPT (GPT-4/Web):** "As of my last update, Ghana's central bank has issued warnings about cryptocurrencies but formal regulation details might not be in the main budget statement itself, possibly in other financial policy documents. There wasn't a major focus on it in typical budget reports..." (Might speculate or provide general knowledge, potentially hallucinating if it tries to invent a stance).
            * **Analysis:** RAG excels at confirming the *absence* of information within the specific document, reducing hallucination risk. ChatGPT might try to provide a broader (but potentially inaccurate for this specific document) answer.

        **Conclusion:** The RAG system is superior for answering questions *specifically* about the content of the provided document (2024 Budget). It is less prone to hallucination on topics outside the document's scope. ChatGPT has broader world knowledge and can synthesize information but lacks the precise, indexed knowledge of this specific PDF and may provide answers that are plausible but not directly supported by *this* document's text. RAG's limitation is its knowledge boundary – it only knows what's in the PDF.
        """)

# --- Sidebar Info ---
st.sidebar.markdown("---")
st.sidebar.info("ML & AI Explorer v1.0")