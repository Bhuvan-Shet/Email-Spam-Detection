
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import nltk
import streamlit as st
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Email Spam Detection")
page = st.sidebar.radio("Navigation", ["Home", "Data Exploration", "Model Training", "Model Evaluation", "Spam Predictor"])

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'cv' not in st.session_state:
    st.session_state.cv = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'Y_train' not in st.session_state:
    st.session_state.Y_train = None
if 'Y_test' not in st.session_state:
    st.session_state.Y_test = None
if 'corpus' not in st.session_state:
    st.session_state.corpus = None

# Function to preprocess text
def preprocess_text(message):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

# Function to load data
def load_data():
    try:
        # Show file uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Read the data
            spam = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            
            # Process the dataset
            spam = spam[['v1', 'v2']]
            spam.columns = ['label', 'message']
            
            st.session_state.data_loaded = True
            return spam
        
        # If no file is uploaded, provide an option to use sample data
        if st.button("Use Sample Data"):
            # This is just a placeholder - in a real app you'd have sample data embedded or hosted
            sample_data = {
                'label': ['ham', 'spam', 'ham', 'ham', 'spam'],
                'message': [
                    'Go until jurong point, crazy.. Available only in bugis n great world la e buffet...',
                    'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.',
                    'U dun say so early hor... U c already then say...',
                    'Nah I dont think he goes to usf, he lives around here though',
                    'WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!'
                ]
            }
            spam = pd.DataFrame(sample_data)
            st.session_state.data_loaded = True
            return spam
        
        return None
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Process the data and create features
def process_data(spam):
    with st.spinner("Processing data..."):
        # Preprocess the text
        ps = PorterStemmer()
        corpus = []
        for i in range(0, len(spam)):
            review = preprocess_text(spam['message'][i])
            corpus.append(review)
        
        st.session_state.corpus = corpus
        
        # Create bag of words model
        cv = CountVectorizer(max_features=4000)
        X = cv.fit_transform(corpus).toarray()
        Y = pd.get_dummies(spam['label'])
        Y = Y.iloc[:, 1].values
        
        # Save the CountVectorizer for later use
        st.session_state.cv = cv
        
        # Split the dataset
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
        
        # Save to session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.Y_train = Y_train
        st.session_state.Y_test = Y_test
        
        return X_train, X_test, Y_train, Y_test

# Train models
def train_models(X_train, Y_train):
    with st.spinner("Training models..."):
        # Random Forest Classifier
        model1 = RandomForestClassifier()
        model1.fit(X_train, Y_train)
        
        # Decision Tree Classifier
        model2 = DecisionTreeClassifier()
        model2.fit(X_train, Y_train)
        
        # Multinomial Naïve Bayes
        model3 = MultinomialNB()
        model3.fit(X_train, Y_train)
        
        # Save models to session state
        st.session_state.models = {
            "Random Forest Classifier": model1,
            "Decision Tree Classifier": model2,
            "Multinomial Naïve Bayes": model3
        }
        
        return model1, model2, model3

# Evaluate models
# Evaluate models - Fixed version
def evaluate_models(models, X_test, Y_test):
    results = {}
    
    for name, model in models.items():
        pred = model.predict(X_test)
        cm = confusion_matrix(Y_test, pred)
        acc = accuracy_score(Y_test, pred)
        
        # Get classification report as dictionary
        report = classification_report(Y_test, pred, output_dict=True)
        
        # Get report keys that are not summary metrics
        class_keys = [key for key in report.keys() 
                     if key not in ['accuracy', 'macro avg', 'weighted avg']]
        
        # In our dataset, spam is labeled as 1 (or "1")
        # Find the spam class key (could be "1", 1, or other)
        spam_key = None
        for key in class_keys:
            try:
                # If converting to float works, compare it
                if spam_key is None or float(key) > float(spam_key):
                    spam_key = key
            except ValueError:
                # If can't convert to float, skip this key
                continue
        
        # If we couldn't find a numeric key, use the last key (assuming binary classification)
        if spam_key is None and len(class_keys) > 0:
            spam_key = class_keys[-1]
            
        # Create a new result dictionary with explicit spam class identification
        results[name] = {
            "predictions": pred,
            "confusion_matrix": cm,
            "accuracy": acc,
            "report": report,
            "spam_key": spam_key  # Store the identified spam key
        }
    
    return results

# Save models
def save_models(models):
    for name, model in models.items():
        filename = f"{name.replace(' ', '_')}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
        # Offer download option
        with open(filename, 'rb') as file:
            st.download_button(
                label=f"Download {name} Model",
                data=file,
                file_name=filename,
                mime="application/octet-stream"
            )

# Home page
if page == "Home":
    st.title("📧 Email Spam Detection System")
    
    st.markdown("""
    Welcome to the Email Spam Detection System! This application helps you:
    
    1. Explore your email dataset
    2. Train machine learning models to detect spam
    3. Evaluate model performance
    4. Predict whether new emails are spam or not
    
    ### How to use this app:
    
    1. Navigate through the different sections using the sidebar
    2. Start by uploading your spam dataset in the Data Exploration section
    3. Train models and evaluate their performance
    4. Use the trained models to predict spam emails
    
    ### Technology Used:
    
    - **Natural Language Processing (NLP)** for text preprocessing
    - **Machine Learning Models**: Random Forest, Decision Tree, and Multinomial Naïve Bayes
    - **Streamlit** for the interactive web interface
    
    Get started by navigating to the **Data Exploration** section!
    """)
    
    st.image("emailspam.jpg", caption="Email Spam Detection")

# Data Exploration page
elif page == "Data Exploration":
    st.title("Data Exploration")
    
    # Load data
    spam = load_data()
    
    if spam is not None:
        # Display basic info
        st.subheader("Dataset Overview")
        st.write(f"Dataset Shape: {spam.shape}")
        
        # Display first few rows
        st.subheader("First 5 Rows")
        st.dataframe(spam.head())
        
        # Check for null values
        st.subheader("Null Values")
        null_counts = spam.isnull().sum()
        st.write(null_counts)
        
        # Distribution of spam vs ham
        st.subheader("Distribution of Spam vs Ham")
        
        col1, col2 = st.columns(2)
        
        with col1:
            distribution = spam['label'].value_counts().reset_index()
            distribution.columns = ['Label', 'Count']
            st.dataframe(distribution)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x='label', data=spam, palette='viridis')
            plt.title('Distribution of Spam vs Ham')
            plt.xlabel('Label')
            plt.ylabel('Count')
            st.pyplot(fig)
        
        # Message length analysis
        st.subheader("Message Length Analysis")
        
        # Add message length to dataset
        spam['message_length'] = spam['message'].apply(len)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Message Length Statistics")
            st.dataframe(spam.groupby('label')['message_length'].describe())
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='label', y='message_length', data=spam, palette='viridis')
            plt.title('Message Length by Category')
            plt.xlabel('Label')
            plt.ylabel('Message Length')
            st.pyplot(fig)
        
        # Sample messages
        st.subheader("Sample Messages")
        
        tabs = st.tabs(["Spam Samples", "Ham Samples"])
        
        with tabs[0]:
            spam_samples = spam[spam['label'] == 'spam'].head(5)
            for i, row in enumerate(spam_samples.itertuples()):
                st.text_area(f"Spam Example {i+1}", row.message, height=100, key=f"spam_{i}")
        
        with tabs[1]:
            ham_samples = spam[spam['label'] == 'ham'].head(5)
            for i, row in enumerate(ham_samples.itertuples()):
                st.text_area(f"Ham Example {i+1}", row.message, height=100, key=f"ham_{i}")
        
        # Process data button
        if st.button("Process Data for Training"):
            X_train, X_test, Y_train, Y_test = process_data(spam)
            st.success("Data processed successfully! Go to Model Training section.")

# Model Training page
elif page == "Model Training":
    st.title("Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please load and process your data in the Data Exploration section first.")
    else:
        st.write("Train machine learning models on your preprocessed data.")
        
        # Show data split information
        if st.session_state.X_train is not None:
            st.info(f"Training set size: {st.session_state.X_train.shape[0]} samples")
            st.info(f"Test set size: {st.session_state.X_test.shape[0]} samples")
        
        # Train models button
        if st.button("Train Models"):
            if st.session_state.X_train is not None and st.session_state.Y_train is not None:
                model1, model2, model3 = train_models(st.session_state.X_train, st.session_state.Y_train)
                st.success("Models trained successfully! Go to Model Evaluation section.")
            else:
                st.error("Training data not available. Please process your data first.")

# Model Evaluation page
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    
    if not st.session_state.models:
        st.warning("Please train your models in the Model Training section first.")
    else:
        results = evaluate_models(
            st.session_state.models, 
            st.session_state.X_test, 
            st.session_state.Y_test
        )
        
        # Model comparison
        st.subheader("Model Accuracy Comparison")
        
        # Create a DataFrame for comparison
        comparison_data = {
            "Model": [],
            "Accuracy": [],
            "Precision (Spam)": [],
            "Recall (Spam)": [],
            "F1-Score (Spam)": []
        }
        
        for name, result in results.items():
            comparison_data["Model"].append(name)
            comparison_data["Accuracy"].append(result["accuracy"])
              # Corrected lines:
            if "True" in result["report"]:
             comparison_data["Precision (Spam)"].append(result["report"]["True"]["precision"])
             comparison_data["Recall (Spam)"].append(result["report"]["True"]["recall"])
             comparison_data["F1-Score (Spam)"].append(result["report"]["True"]["f1-score"])
            else:
        # Handle the case where "True" is not found.
             comparison_data["Precision (Spam)"].append(0.0)  # Or some other default value
             comparison_data["Recall (Spam)"].append(0.0)
             comparison_data["F1-Score (Spam)"].append(0.0)

        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df["Accuracy"].idxmax()]["Model"]
        st.success(f"Best performing model: {best_model} with accuracy of {comparison_df['Accuracy'].max():.4f}")
        
        # Detailed results
        st.subheader("Detailed Model Evaluation")
        
        for name, result in results.items():
            with st.expander(f"{name} (Accuracy: {result['accuracy']:.4f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Confusion Matrix")
                    cm = result["confusion_matrix"]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=['Ham', 'Spam'],
                        yticklabels=['Ham', 'Spam']
                    )
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix - {name}')
                    st.pyplot(fig)
                
                with col2:
                    st.write("Classification Report")
                    report_df = pd.DataFrame(result["report"]).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
        
        # Save models
        st.subheader("Save Trained Models")
        if st.button("Save Models"):
            save_models(st.session_state.models)
            st.success("Models saved successfully!")

# Spam Predictor page
elif page == "Spam Predictor":
    st.title("Spam Email Predictor")
    
    if not st.session_state.models:
        st.warning("Please train your models in the Model Training section first.")
    else:
        st.write("Test the models with your own email message.")
        
        # Select model
        model_option = st.selectbox(
            "Select a model for prediction",
            list(st.session_state.models.keys())
        )
        
        # Input for email message
        email_message = st.text_area("Enter an email message", height=200)
        
        if st.button("Predict"):
            if email_message:
                # Preprocess the input
                processed_message = preprocess_text(email_message)
                
                # Transform using the CountVectorizer
                message_vector = st.session_state.cv.transform([processed_message]).toarray()
                
                # Make prediction
                model = st.session_state.models[model_option]
                prediction = model.predict(message_vector)[0]
                
                # Display result
                if prediction == 1:
                    st.error("⚠️ This message is classified as SPAM!")
                    
                    # Calculate and display probability if available
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(message_vector)[0][1]
                        st.write(f"Probability of being spam: {probability:.2%}")
                    
                    # Provide explanation
                    st.write("This message was classified as spam because it may contain:")
                    st.write("- Promotional offers or phrases like 'free', 'win', or 'prize'")
                    st.write("- Unusual formatting, excessive punctuation, or ALL CAPS")
                    st.write("- Requests for personal information")
                    st.write("- Urgency indicators or pressure tactics")
                else:
                    st.success("✅ This message is classified as HAM (not spam).")
                    
                    # Calculate and display probability if available
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(message_vector)[0][0]
                        st.write(f"Probability of being legitimate: {probability:.2%}")
            else:
                st.warning("Please enter an email message.")