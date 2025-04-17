import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf # Use tensorflow explicitly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input # Add Input layer for explicit shape
from tensorflow.keras.optimizers import Adam, SGD # Import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback # For progress visualization

import PyPDF2
import io # To handle uploaded file bytes

# LLM RAG Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter # Using langchain for convenience

# Configure page
st.set_page_config(page_title="ML & AI Explorer", layout="wide")

# --- Helper Functions ---

@st.cache_data # Cache data loading and basic processing
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

@st.cache_data # Cache scaling
def scale_data(df, columns_to_scale):
    scaler = StandardScaler()
    df_scaled = df.copy()
    # Check if columns exist and are numeric before scaling
    numeric_cols_to_scale = [col for col in columns_to_scale if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols_to_scale:
        df_scaled[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
        return df_scaled, scaler, numeric_cols_to_scale # Return scaler and scaled columns
    else:
        st.warning("No numeric columns selected or found for scaling.")
        return df, None, []

# Custom Keras callback for Streamlit progress
class StreamlitCallback(Callback):
    def __init__(self, progress_bar, status_text, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.epochs = epochs
        self.epoch_losses = []
        self.epoch_val_losses = []
        self.epoch_acc = []
        self.epoch_val_acc = []
        self.chart_loss = st.line_chart()
        self.chart_acc = st.line_chart()


    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        status_message = f"Epoch {epoch + 1}/{self.epochs} - loss: {logs['loss']:.4f}"
        if 'accuracy' in logs: status_message += f" - accuracy: {logs['accuracy']:.4f}"
        if 'val_loss' in logs: status_message += f" - val_loss: {logs['val_loss']:.4f}"
        if 'val_accuracy' in logs: status_message += f" - val_accuracy: {logs['val_accuracy']:.4f}"
        self.status_text.text(status_message)

        # Append metrics for plotting
        self.epoch_losses.append(logs.get('loss'))
        self.epoch_val_losses.append(logs.get('val_loss'))
        self.epoch_acc.append(logs.get('accuracy'))
        self.epoch_val_acc.append(logs.get('val_accuracy'))

        # Update charts (only plot if validation data exists for val metrics)
        loss_data = {"Training Loss": self.epoch_losses}
        if self.epoch_val_losses and self.epoch_val_losses[0] is not None:
             loss_data["Validation Loss"] = self.epoch_val_losses
        self.chart_loss.line_chart(pd.DataFrame(loss_data))

        acc_data = {}
        if self.epoch_acc and self.epoch_acc[0] is not None:
            acc_data["Training Accuracy"] = self.epoch_acc
        if self.epoch_val_acc and self.epoch_val_acc[0] is not None:
            acc_data["Validation Accuracy"] = self.epoch_val_acc
        if acc_data: # Only plot accuracy if it's available
             self.chart_acc.line_chart(pd.DataFrame(acc_data))

# --- LLM RAG Setup ---

# Cache the expensive model loading
@st.cache_resource
def load_llm_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    st.info(f"Loading LLM: {model_name}... This may take a while and require significant RAM/Disk space.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, # Use float16 to save memory if possible
            device_map="auto", # Automatically use GPU if available
        )
        # Set pad token if missing (common for some models like Mistral)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        st.success("LLM loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading LLM: {e}. Ensure you have enough resources and dependencies installed.")
        st.stop() # Stop execution if model loading fails

@st.cache_resource
def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    st.info(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()

# Function to build FAISS index (cache resource to avoid rebuilding)
# Use session state or a more robust caching mechanism if index needs persistence across uploads
@st.cache_resource(show_spinner=False)
def build_faiss_index(text_chunks, _embedding_model):
    if not text_chunks:
        return None, None
    with st.spinner("Embedding text chunks and building index..."):
        try:
            embeddings = _embedding_model.encode(text_chunks, convert_to_tensor=False) # Get numpy arrays
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension) # Using simple L2 distance index
            index.add(np.array(embeddings).astype('float32')) # FAISS requires float32
            return index, text_chunks
        except Exception as e:
            st.error(f"Error building FAISS index: {e}")
            return None, None

def search_faiss_index(query, _embedding_model, index, text_chunks, k=3):
    if index is None or query == "":
        return [], []
    with st.spinner("Searching for relevant context..."):
        try:
            query_embedding = _embedding_model.encode([query], convert_to_tensor=False)[0]
            distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
            relevant_chunks = [text_chunks[i] for i in indices[0]]
            similarity_scores = [1 / (1 + d) for d in distances[0]] # Simple similarity score from distance
            return relevant_chunks, similarity_scores
        except Exception as e:
            st.error(f"Error searching FAISS index: {e}")
            return [], []

def generate_llm_response(tokenizer, model, prompt):
     with st.spinner("Generating response from LLM..."):
        try:
            # Create pipeline for text generation for easier handling
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=250, # Limit response length
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
             )
            # Mistral Instruct format often uses [INST] [/INST]
            formatted_prompt = f"[INST] {prompt} [/INST]"
            result = pipe(formatted_prompt)
            # Extract only the generated text after the prompt
            # The exact extraction logic might depend slightly on the pipeline version/model behavior
            full_response = result[0]['generated_text']
            # Find the end of the prompt marker to isolate the answer
            answer_start_index = full_response.rfind("[/INST]") + len("[/INST]")
            answer = full_response[answer_start_index:].strip()

            return answer
        except Exception as e:
            st.error(f"Error during LLM inference: {e}")
            return "Sorry, I encountered an error generating the response."

# --- Streamlit App UI ---

st.sidebar.title("ML & AI Explorer")
task = st.sidebar.radio("Choose a Task", ["Regression", "Clustering", "Neural Network", "LLM Q&A"])

# Initialize session state for RAG components if they don't exist
if 'rag_index' not in st.session_state:
    st.session_state.rag_index = None
if 'rag_chunks' not in st.session_state:
    st.session_state.rag_chunks = None
if 'processed_pdf_name' not in st.session_state:
    st.session_state.processed_pdf_name = None

# --- Regression Section ---
if task == "Regression":
    st.title("📈 Regression Explorer")
    st.markdown("Build a simple linear regression model to predict a continuous variable.")

    uploaded_file = st.file_uploader("1. Upload CSV for Regression", type="csv")

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Dataset Preview (First 5 Rows):")
            st.dataframe(df.head())

            # Select features and target
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            non_numeric_columns = df.select_dtypes(exclude=np.number).columns.tolist()

            if not numeric_columns:
                 st.warning("No numeric columns found in the uploaded CSV. Regression requires numeric features and target.")
                 st.stop()

            st.subheader("2. Select Features and Target Variable")
            target_col = st.selectbox("Select Target Column (must be numeric)", options=numeric_columns)
            feature_cols = st.multiselect("Select Feature Columns (numeric recommended)", options=numeric_columns, default=[col for col in numeric_columns if col != target_col])

            if not target_col:
                st.warning("Please select a target column.")
            elif not feature_cols:
                st.warning("Please select at least one feature column.")
            else:
                X = df[feature_cols]
                y = df[target_col]

                # Preprocessing Option
                st.subheader("3. Preprocessing (Optional)")
                scale_features = st.checkbox("Scale Features (StandardScaler)")
                scaler = None # Initialize scaler
                if scale_features:
                    X_scaled, scaler, scaled_cols = scale_data(X, feature_cols)
                    if scaler:
                        X = X_scaled # Use scaled data
                        st.write("Features scaled:", scaled_cols)


                st.subheader("4. Train Model and View Results")
                if st.button("Train Linear Regression Model"):
                    with st.spinner("Training Model..."):
                        model = LinearRegression()
                        model.fit(X, y)
                        predictions = model.predict(X)

                        mae = mean_absolute_error(y, predictions)
                        r2 = r2_score(y, predictions)

                        st.write("**Model Performance:**")
                        st.write(f"- Mean Absolute Error (MAE): ${mae:.2f}$")
                        st.write(f"- R² Score: ${r2:.4f}$")

                        # Visualization
                        st.write("**Actual vs. Predicted Values**")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y, predictions, alpha=0.6, label='Data points')
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2, label='Ideal Line (y=x)') # Add y=x line
                        ax.set_xlabel("Actual Values")
                        ax.set_ylabel("Predicted Values")
                        ax.set_title("Actual vs. Predicted Values")
                        ax.legend()
                        st.pyplot(fig)

                        # Store model and scaler in session state if needed for prediction later
                        st.session_state['regression_model'] = model
                        st.session_state['regression_scaler'] = scaler
                        st.session_state['regression_features'] = feature_cols
                        st.session_state['regression_target'] = target_col
                        st.success("Model trained successfully!")

                # Prediction Section
                if 'regression_model' in st.session_state:
                    st.subheader("5. Make Predictions on Custom Data")
                    st.write(f"Enter values for the features: {', '.join(st.session_state['regression_features'])}")
                    input_data = {}
                    cols = st.columns(len(st.session_state['regression_features']))
                    for i, col in enumerate(st.session_state['regression_features']):
                         # Use number_input, ensure correct type handling might be needed
                        input_data[col] = cols[i].number_input(f"{col}", value=float(df[col].mean()), key=f"pred_{col}")

                    if st.button("Predict Value"):
                        input_df = pd.DataFrame([input_data])
                        scaler = st.session_state.get('regression_scaler', None) # Retrieve scaler
                        model = st.session_state['regression_model']

                        # Apply scaling if it was used during training
                        if scaler:
                            # Only scale the features that were originally scaled
                            scaled_pred_cols = [col for col in st.session_state['regression_features'] if col in scaler.feature_names_in_]
                            if scaled_pred_cols:
                                input_df[scaled_pred_cols] = scaler.transform(input_df[scaled_pred_cols])
                            else:
                                st.warning("Scaler exists but no matching columns found in input for scaling.")


                        try:
                            pred = model.predict(input_df)[0]
                            st.success(f"Predicted {st.session_state['regression_target']}: ${pred:.2f}$")
                        except Exception as e:
                             st.error(f"Prediction error: {e}")
                             st.warning("Ensure input values are numeric and match the model's expectations.")


# --- Clustering Section ---
elif task == "Clustering":
    st.title("📊 Clustering Explorer")
    st.markdown("Group data points into clusters using the K-Means algorithm.")

    uploaded_file = st.file_uploader("1. Upload CSV for Clustering", type="csv")

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Dataset Preview (First 5 Rows):")
            st.dataframe(df.head())

            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_columns:
                 st.warning("No numeric columns found. Clustering typically requires numeric features.")
                 st.stop()

            st.subheader("2. Select Features for Clustering")
            cluster_cols = st.multiselect("Select Features (numeric recommended)", options=numeric_columns, default=numeric_columns[:min(len(numeric_columns), 2)]) # Default to first 2

            if not cluster_cols:
                st.warning("Please select at least one feature for clustering.")
            else:
                X = df[cluster_cols]

                # Preprocessing Option
                st.subheader("3. Preprocessing (Optional)")
                scale_features_cluster = st.checkbox("Scale Features (StandardScaler)")
                scaler_cluster = None
                if scale_features_cluster:
                    X_scaled, scaler_cluster, scaled_cluster_cols = scale_data(X, cluster_cols)
                    if scaler_cluster:
                        X = X_scaled
                        st.write("Features scaled:", scaled_cluster_cols)

                st.subheader("4. Perform Clustering")
                n_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3, step=1)

                if st.button("Run K-Means Clustering"):
                     with st.spinner("Performing Clustering..."):
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init explicitly
                        # Ensure X only contains numeric data before fitting
                        X_numeric = X.select_dtypes(include=np.number)
                        if X_numeric.empty:
                            st.error("Selected features do not contain numeric data suitable for K-Means.")
                            st.stop()

                        cluster_labels = kmeans.fit_predict(X_numeric)
                        centroids = kmeans.cluster_centers_

                        # Add cluster labels to the original dataframe (or scaled if scaling applied)
                        df_clustered = df.copy() # Start with original
                        df_clustered['Cluster'] = cluster_labels
                        st.session_state['clustered_df'] = df_clustered # Save for download

                        st.success(f"K-Means finished. {n_clusters} clusters found.")

                        # --- Visualization ---
                        st.subheader("5. Visualize Clusters")
                        plot_data = X_numeric.copy() # Use the data that was actually clustered (potentially scaled)
                        plot_data['Cluster'] = cluster_labels

                        # If more than 2 features, use PCA for visualization
                        pca = None
                        if plot_data.shape[1] > 3: # More than 2 features + 1 Cluster col
                            st.write("Using PCA to reduce dimensions for 2D visualization.")
                            pca = PCA(n_components=2)
                            plot_data_reduced = pca.fit_transform(plot_data.drop('Cluster', axis=1))
                            plot_df = pd.DataFrame(plot_data_reduced, columns=['PC1', 'PC2'])
                            plot_df['Cluster'] = cluster_labels
                            # Transform centroids
                            centroids_reduced = pca.transform(centroids)
                        elif plot_data.shape[1] == 3: # Exactly 2 features + 1 Cluster col
                            plot_df = plot_data
                            plot_df.columns = [cluster_cols[0], cluster_cols[1], 'Cluster'] # Use original names
                            centroids_reduced = centroids # Centroids are already in 2D
                        else: # 1 feature + 1 Cluster col
                            st.write("Plotting cluster distribution for the single selected feature.")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.stripplot(data=plot_df, x=plot_df.columns[0], y='Cluster', hue='Cluster', orient='h', palette='viridis', jitter=True, ax=ax)
                            ax.scatter(centroids[:, 0], range(n_clusters), marker='X', s=200, c='red', label='Centroids')
                            ax.set_title(f"Cluster Distribution for {plot_df.columns[0]}")
                            ax.set_xlabel(plot_df.columns[0])
                            ax.set_ylabel("Cluster")
                            ax.legend()
                            st.pyplot(fig)
                            plot_df = None # Indicate no 2D plot generated

                        # 2D Scatter plot (if dimensions allow)
                        if plot_df is not None and centroids_reduced is not None:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.scatterplot(data=plot_df, x=plot_df.columns[0], y=plot_df.columns[1], hue='Cluster', palette='viridis', s=50, alpha=0.7, ax=ax)
                            # Plot centroids
                            ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='X', s=200, c='red', label='Centroids')
                            ax.set_title(f'K-Means Clustering (K={n_clusters})')
                            ax.set_xlabel(plot_df.columns[0])
                            ax.set_ylabel(plot_df.columns[1])
                            ax.legend(title='Cluster')
                            st.pyplot(fig)

                # Download Button
                if 'clustered_df' in st.session_state:
                    st.subheader("6. Download Results")
                    csv_data = st.session_state['clustered_df'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Clustered Data as CSV",
                        data=csv_data,
                        file_name=f'clustered_data_{uploaded_file.name}.csv',
                        mime='text/csv',
                    )


# --- Neural Network Section ---
elif task == "Neural Network":
    st.title("🧠 Neural Network Trainer")
    st.markdown("Build and train a simple Feedforward Neural Network for classification.")

    uploaded_file = st.file_uploader("1. Upload CSV for Classification", type="csv")

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Dataset Preview (First 5 Rows):")
            st.dataframe(df.head())

            st.subheader("2. Select Features and Target Variable")
            all_columns = df.columns.tolist()
            target_col_nn = st.selectbox("Select Target Column (Categorical/Label)", options=all_columns)
            # Exclude target column from potential features by default
            default_features = [col for col in df.select_dtypes(include=np.number).columns.tolist() if col != target_col_nn]
            feature_cols_nn = st.multiselect("Select Feature Columns (numeric)", options=df.select_dtypes(include=np.number).columns.tolist(), default=default_features)


            if not target_col_nn:
                st.warning("Please select a target column.")
            elif not feature_cols_nn:
                st.warning("Please select at least one numeric feature column.")
            else:
                X = df[feature_cols_nn]
                # Target processing
                try:
                    # Attempt to convert target to categorical codes
                    y_labels = df[target_col_nn].astype('category')
                    y_codes = y_labels.cat.codes
                    num_classes = len(y_labels.cat.categories)
                    y = to_categorical(y_codes, num_classes=num_classes)
                    st.write(f"Target variable '{target_col_nn}' processed into {num_classes} classes.")
                except Exception as e:
                    st.error(f"Error processing target column '{target_col_nn}': {e}")
                    st.warning("Ensure the target column is suitable for classification (e.g., categorical or integer labels).")
                    st.stop()

                # Preprocessing Option
                st.subheader("3. Preprocessing & Data Split")
                scale_features_nn = st.checkbox("Scale Features (StandardScaler)")
                scaler_nn = None
                if scale_features_nn:
                    X_scaled, scaler_nn, scaled_nn_cols = scale_data(X, feature_cols_nn)
                    if scaler_nn:
                         X = X_scaled
                         st.write("Features scaled:", scaled_nn_cols)

                test_size = st.slider("Validation Split Ratio", 0.1, 0.5, 0.2, 0.05)
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if num_classes > 1 else None) # Stratify for classification

                st.write(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

                st.subheader("4. Configure Neural Network")
                # Hyperparameters
                hidden_layer_1_nodes = st.slider("Nodes in Hidden Layer 1", 16, 128, 64, 8)
                hidden_layer_2_nodes = st.slider("Nodes in Hidden Layer 2", 8, 64, 32, 8)
                activation_fn = st.selectbox("Activation Function", ['relu', 'sigmoid', 'tanh'], index=0)
                optimizer_name = st.selectbox("Optimizer", ['adam', 'sgd'], index=0)
                lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 0.0005, format="%.4f")
                epochs = st.slider("Epochs", 5, 100, 15, 5)

                # Select optimizer based on choice
                if optimizer_name == 'adam':
                    optimizer = Adam(learning_rate=lr)
                elif optimizer_name == 'sgd':
                    optimizer = SGD(learning_rate=lr)
                else: # Default just in case
                    optimizer = Adam(learning_rate=lr)

                st.subheader("5. Train the Model")
                if st.button("Start Training"):
                    # Build Model
                    model = Sequential([
                        Input(shape=(X_train.shape[1],)), # Explicit input layer
                        Dense(hidden_layer_1_nodes, activation=activation_fn),
                        Dense(hidden_layer_2_nodes, activation=activation_fn),
                        Dense(num_classes, activation='softmax') # Softmax for multi-class
                    ])
                    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                    model.summary(print_fn=st.text) # Print model summary to Streamlit

                    # Prepare for real-time plotting
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    st_callback = StreamlitCallback(progress_bar, status_text, epochs)


                    with st.spinner("Training in progress..."):
                        history = model.fit(X_train, y_train,
                                          validation_data=(X_val, y_val),
                                          epochs=epochs,
                                          callbacks=[st_callback],
                                          verbose=0) # Use callback for output

                    st.success("Training finished!")
                    # Display final metrics from history object as well
                    final_train_acc = history.history['accuracy'][-1]
                    final_val_acc = history.history['val_accuracy'][-1]
                    st.write(f"**Final Training Accuracy:** {final_train_acc:.4f}")
                    st.write(f"**Final Validation Accuracy:** {final_val_acc:.4f}")


                    st.session_state['nn_model'] = model
                    st.session_state['nn_scaler'] = scaler_nn
                    st.session_state['nn_features'] = feature_cols_nn
                    st.session_state['nn_target_classes'] = y_labels.cat.categories.tolist() # Save class names


                # Prediction Section for NN
                if 'nn_model' in st.session_state:
                    st.subheader("6. Make Predictions on Custom Data")
                    st.write(f"Enter values for the features: {', '.join(st.session_state['nn_features'])}")
                    input_data_nn = {}
                    cols_nn = st.columns(len(st.session_state['nn_features']))
                    for i, col in enumerate(st.session_state['nn_features']):
                        input_data_nn[col] = cols_nn[i].number_input(f"{col}", value=float(df[col].mean()), key=f"pred_nn_{col}")

                    if st.button("Predict Class"):
                        input_df_nn = pd.DataFrame([input_data_nn])
                        scaler_nn = st.session_state.get('nn_scaler', None)
                        model_nn = st.session_state['nn_model']
                        class_names = st.session_state['nn_target_classes']

                        # Apply scaling if used during training
                        if scaler_nn:
                            scaled_pred_cols_nn = [col for col in st.session_state['nn_features'] if col in scaler_nn.feature_names_in_]
                            if scaled_pred_cols_nn:
                                input_df_nn[scaled_pred_cols_nn] = scaler_nn.transform(input_df_nn[scaled_pred_cols_nn])
                            else:
                                st.warning("Scaler exists but no matching columns found in input for scaling.")


                        try:
                            pred_proba = model_nn.predict(input_df_nn)
                            pred_class_index = np.argmax(pred_proba, axis=1)[0]
                            pred_class_name = class_names[pred_class_index]
                            pred_confidence = pred_proba[0][pred_class_index]
                            st.success(f"Predicted Class: **{pred_class_name}** (Confidence: {pred_confidence:.2f})")
                            st.write("Prediction Probabilities per Class:")
                            st.dataframe(pd.DataFrame(pred_proba, columns=class_names))
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                            st.warning("Ensure input values are numeric and match the model's expectations.")

                    st.subheader("OR Upload Test CSV for Prediction")
                    test_file = st.file_uploader("Upload Test CSV", type="csv", key="nn_test_upload")
                    if test_file:
                         test_df = load_data(test_file)
                         if test_df is not None:
                             st.write("Test Data Preview:")
                             st.dataframe(test_df.head())
                             # Ensure test data has the necessary feature columns
                             if not all(col in test_df.columns for col in st.session_state['nn_features']):
                                 st.error(f"Test CSV must contain the feature columns: {st.session_state['nn_features']}")
                             else:
                                 X_test_custom = test_df[st.session_state['nn_features']]
                                 scaler_nn = st.session_state.get('nn_scaler', None)
                                 model_nn = st.session_state['nn_model']
                                 class_names = st.session_state['nn_target_classes']

                                 # Apply scaling if used during training
                                 if scaler_nn:
                                     scaled_pred_cols_nn = [col for col in st.session_state['nn_features'] if col in scaler_nn.feature_names_in_]
                                     if scaled_pred_cols_nn:
                                         X_test_custom[scaled_pred_cols_nn] = scaler_nn.transform(X_test_custom[scaled_pred_cols_nn])

                                 try:
                                     pred_proba_test = model_nn.predict(X_test_custom)
                                     pred_class_indices_test = np.argmax(pred_proba_test, axis=1)
                                     pred_class_names_test = [class_names[i] for i in pred_class_indices_test]

                                     results_df = test_df.copy()
                                     results_df['Predicted_Class'] = pred_class_names_test
                                     results_df['Confidence'] = np.max(pred_proba_test, axis=1)

                                     st.write("Predictions on Test CSV:")
                                     st.dataframe(results_df)

                                     # Option to download predictions
                                     csv_pred_data = results_df.to_csv(index=False).encode('utf-8')
                                     st.download_button(
                                         label="Download Test Predictions as CSV",
                                         data=csv_pred_data,
                                         file_name=f'predictions_{test_file.name}.csv',
                                         mime='text/csv',
                                     )

                                 except Exception as e:
                                     st.error(f"Prediction error on test file: {e}")


# --- LLM Q&A Section ---
else: # LLM Task
    st.title("💬 Large Language Model (LLM) Q&A - RAG Approach")
    st.markdown("""
    Ask questions based on the content of an uploaded PDF document.
    This section uses the **Retrieval-Augmented Generation (RAG)** approach with the
    `mistralai/Mistral-7B-Instruct-v0.1` model.
    """)

    # --- LLM Documentation Expander ---
    with st.expander("About this LLM Section (RAG Approach)"):
        st.subheader("1. Dataset & Model")
        st.markdown("""
        * **Dataset:** You can upload any PDF document. The content will be extracted and used as the knowledge base.
        * **LLM Model:** `mistralai/Mistral-7B-Instruct-v0.1` (via Hugging Face Transformers). A powerful open-source instruction-tuned model.
        * **Embedding Model:** `all-MiniLM-L6-v2` (via Sentence Transformers). Used to convert text chunks and questions into vector representations for similarity search.
        * **Vector Store:** FAISS (Facebook AI Similarity Search). An efficient library for searching in large sets of vectors.
        """)

        st.subheader("2. RAG Architecture")
        # Placeholder for Architecture Diagram/Description
        st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3fKUdy3T7MyS2fM_qVZb4Q.png", caption="Generic RAG Architecture Diagram", use_column_width=True) # Replace with your specific diagram/link if available
        st.markdown("""
        **Flow:**
        1.  **Load & Chunk:** The uploaded PDF is loaded, text extracted, and split into smaller, manageable chunks.
        2.  **Embed & Index:** Each text chunk is converted into a numerical vector (embedding) using the embedding model. These embeddings are stored in a FAISS index for fast retrieval.
        3.  **Query:** The user asks a question.
        4.  **Embed Query:** The user's question is also embedded using the same embedding model.
        5.  **Retrieve:** The FAISS index is searched to find the text chunks whose embeddings are most similar to the question embedding (these are the most relevant context chunks).
        6.  **Prompt:** The retrieved context chunks and the original question are combined into a prompt for the LLM.
        7.  **Generate:** The LLM receives the prompt and generates an answer based *only* on the provided context and its internal knowledge conditioned by the instruction format.
        8.  **Response:** The generated answer is displayed to the user.
        """)

        st.subheader("3. Methodology")
        st.markdown("""
        * **PDF Parsing:** Uses the `PyPDF2` library to extract text content page by page.
        * **Text Splitting:** Employs `RecursiveCharacterTextSplitter` (from LangChain, though could be done manually) to break down the text. This tries to keep related sentences/paragraphs together, splitting recursively based on characters like newlines and spaces. Chunk size and overlap can be adjusted.
        * **Embedding:** Leverages the pre-trained `all-MiniLM-L6-v2` Sentence Transformer model, known for its balance of speed and performance in generating meaningful sentence/paragraph embeddings.
        * **Vector Storage/Retrieval:** Uses a simple FAISS `IndexFlatL2`, which performs an exhaustive L2 (Euclidean) distance search. For larger datasets, more sophisticated FAISS indices (like `IndexIVFFlat`) could be used for faster, approximate searches.
        * **Prompt Engineering:** A specific prompt template (`[INST] Context: {context}\n\nQuestion: {question} [/INST]`) is used to guide the Mistral Instruct model to answer based on the provided context.
        * **LLM Inference:** The `transformers` library pipeline handles tokenization, model inference (using float16 and device mapping for efficiency), and generation parameters (like `max_new_tokens`, `temperature`, `top_p`) to control the output length and creativity.
        """)

        st.subheader("4. Evaluation & Comparison with ChatGPT")
        st.markdown("""
        *(This section would contain your manual analysis)*

        **Evaluation Approach:**
        1.  Define a set of diverse questions relevant to a sample document (e.g., the Academic City policy PDF or the Ghana Budget PDF). Include questions requiring direct fact retrieval, summarization, and potentially some synthesis (though RAG is best at retrieval).
        2.  Run these questions through **this Streamlit application**. Record the answers and note the retrieved context (if possible to expose) and similarity scores.
        3.  Run the *same questions* through **ChatGPT (e.g., GPT-3.5 or GPT-4)**. If possible, provide ChatGPT with the *same document text* as context (copy-paste or upload if using features like the Code Interpreter/Advanced Data Analysis).
        4.  **Compare:**
            * **Accuracy/Correctness:** How factually correct are the answers from each system based *only* on the document?
            * **Completeness:** Does the answer fully address the question?
            * **Conciseness:** Is the answer direct or overly verbose?
            * **Hallucination:** Does either system generate information *not* present in the provided document? (RAG is generally less prone to hallucination *if* the prompt strictly guides it to use the context).
            * **Relevance:** (For RAG) How relevant was the retrieved context to the question? (Use similarity scores as a proxy).
            * **Speed:** Subjective or measured time to get an answer.

        **Example Comparison Points (Hypothetical):**
        * *Mistral-RAG might be better at sticking strictly to the provided text, reducing hallucination risk.*
        * *ChatGPT might provide more fluent or synthesized answers, potentially going beyond the strict text (which could be good or bad depending on the need).*
        * *Mistral-RAG's quality heavily depends on the retrieval step; if relevant context isn't found, the answer might be poor or generic.*
        * *ChatGPT might refuse to answer based on uploaded documents if its safety filters trigger, while the local RAG system has more controllable behavior.*
        * *Resource Usage: The local RAG system requires significant local compute, while ChatGPT is cloud-based.*
        """)

    # --- LLM Application ---
    st.divider()

    # Load models (cached)
    tokenizer, model = load_llm_model()
    embedding_model = load_embedding_model()

    st.subheader("1. Upload PDF Document")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

    # Process PDF only if a new one is uploaded or none is processed
    if uploaded_pdf is not None:
        # Check if it's a new file
        if uploaded_pdf.name != st.session_state.get('processed_pdf_name', None):
            st.info(f"Processing uploaded PDF: {uploaded_pdf.name}")
            # Reset previous index/chunks if a new file is uploaded
            st.session_state.rag_index = None
            st.session_state.rag_chunks = None
            try:
                bytes_data = uploaded_pdf.getvalue()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
                full_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

                if not full_text.strip():
                     st.warning("Could not extract text from the PDF. It might be image-based or protected.")
                else:
                    # Chunk the text
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, # Max characters per chunk
                        chunk_overlap=150, # Overlap between chunks
                        length_function=len,
                    )
                    text_chunks = text_splitter.split_text(full_text)

                    if not text_chunks:
                        st.warning("Text extracted but could not be split into chunks.")
                    else:
                         # Build FAISS index (uses caching)
                         st.session_state.rag_index, st.session_state.rag_chunks = build_faiss_index(text_chunks, embedding_model)
                         if st.session_state.rag_index is not None:
                             st.session_state.processed_pdf_name = uploaded_pdf.name # Mark as processed
                             st.success(f"PDF '{uploaded_pdf.name}' processed. Found {len(text_chunks)} text chunks. Ready for questions.")
                         else:
                             st.error("Failed to build the search index for the PDF.")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.session_state.processed_pdf_name = None # Reset on error
                st.session_state.rag_index = None
                st.session_state.rag_chunks = None
        # else: PDF already processed, do nothing

    # Display Q&A section only if a PDF has been successfully processed
    if st.session_state.get('rag_index') is not None and st.session_state.get('rag_chunks') is not None:
        st.subheader(f"2. Ask Questions about '{st.session_state.processed_pdf_name}'")
        question = st.text_input("Enter your question:", key="llm_question")

        if question:
             # 1. Retrieve relevant context
             relevant_chunks, similarity_scores = search_faiss_index(question, embedding_model, st.session_state.rag_index, st.session_state.rag_chunks, k=3) # Retrieve top 3 chunks

             if not relevant_chunks:
                 st.warning("Could not find relevant context in the document for your question.")
             else:
                 # Display retrieved context and scores (optional)
                 with st.expander("View Retrieved Context (Similarity Score)"):
                     for chunk, score in zip(relevant_chunks, similarity_scores):
                         st.markdown(f"--- (Score: {score:.4f}) ---")
                         st.caption(chunk)


                 # 2. Construct the prompt
                 context_str = "\n\n".join(relevant_chunks)
                 prompt = f"""Based *only* on the following context, please answer the question. If the context doesn't contain the answer, say 'The document does not contain information on this topic'.
Context:
{context_str}

Question: {question}
"""
                 # 3. Generate response
                 answer = generate_llm_response(tokenizer, model, prompt)

                 # 4. Display response
                 st.subheader("Answer:")
                 st.markdown(answer) # Use markdown for potentially formatted LLM output

    elif uploaded_pdf is None:
        st.info("Please upload a PDF document to enable the Q&A feature.")
    # Implicitly handles the case where PDF upload failed or text extraction failed