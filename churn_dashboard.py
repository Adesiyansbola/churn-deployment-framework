# churn_dashboard.py
"""
Streamlit dashboard for customer churn prediction.
This is the interactive frontend for the proof-of-concept.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
This interactive dashboard predicts customer churn risk using machine learning. 
Upload your customer data or use the sample generator to see predictions.
""")

# Load the trained model and feature names
@st.cache_resource
def load_model():
    """Load the trained model and feature names"""
    try:
        model = joblib.load('churn_prediction_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, feature_names = load_model()

# Sidebar for navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose an option:",
    ["📁 Upload Data", "🎲 Generate Sample Data", "ℹ️ About"]
)

if app_mode == "📁 Upload Data":
    st.header("Upload Customer Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with customer data", 
        type="csv"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded data
            new_data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Display basic info
            st.subheader("Data Preview")
            st.dataframe(new_data.head())
            
            st.write(f"**Shape:** {new_data.shape}")
            st.write(f"**Columns:** {list(new_data.columns)}")
            
            # Check if we have the required columns
            required_cols = ['customer_id', 'tenure', 'support_calls', 'subscription_type', 
                           'country', 'monthly_usage', 'avg_session_duration', 
                           'feature_1_usage', 'feature_2_usage']
            
            if all(col in new_data.columns for col in required_cols):
                # Preprocess the new data (same as training)
                processed_data = new_data.copy()
                
                # One-hot encoding for categorical variables
                processed_data = pd.get_dummies(processed_data, 
                                              columns=['subscription_type', 'country'], 
                                              drop_first=True)
                
                # Feature engineering (same as training)
                processed_data['support_intensity'] = processed_data['support_calls'] / processed_data['tenure'].clip(1)
                processed_data['total_usage'] = processed_data['monthly_usage'] * processed_data['tenure']
                processed_data['engagement_score'] = processed_data['avg_session_duration'] * processed_data['monthly_usage']
                
                # Ensure we have all the features the model expects
                for feature in feature_names:
                    if feature not in processed_data.columns:
                        processed_data[feature] = 0  # Add missing features with default value
                
                # Reorder columns to match training data
                processed_data = processed_data[feature_names]
                
                # Make predictions
                churn_probs = model.predict_proba(processed_data)[:, 1]
                
                # Add predictions to original data
                results_df = new_data.copy()
                results_df['churn_probability'] = churn_probs
                results_df['churn_risk'] = results_df['churn_probability'].apply(
                    lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
                )
                
                # Display results
                st.subheader("📈 Prediction Results")
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Customers", len(results_df))
                col2.metric("High Risk Customers", 
                           len(results_df[results_df['churn_risk'] == 'High']))
                col3.metric("Average Churn Probability", 
                           f"{results_df['churn_probability'].mean():.2%}")
                
                # Interactive results table
                st.dataframe(
                    results_df[['customer_id', 'churn_probability', 'churn_risk']]
                    .sort_values('churn_probability', ascending=False)
                    .style.format({'churn_probability': '{:.2%}'})
                    .background_gradient(subset=['churn_probability'], cmap='Reds'),
                    height=300
                )
                
                # Risk distribution chart
                st.subheader("Risk Distribution")
                risk_counts = results_df['churn_risk'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                risk_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax)
                ax.set_title('Customer Churn Risk Distribution')
                ax.set_ylabel('Number of Customers')
                plt.xticks(rotation=0)
                st.pyplot(fig)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.error(f"Missing required columns. Please ensure your CSV contains: {required_cols}")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif app_mode == "🎲 Generate Sample Data":
    st.header("Generate Sample Customer Data")
    
    num_samples = st.slider("Number of samples to generate", 5, 100, 20)
    
    if st.button("Generate Sample Data"):
        # Generate sample data similar to our training data
        sample_data = pd.DataFrame({
            'customer_id': range(1, num_samples + 1),
            'tenure': np.random.randint(1, 72, num_samples),
            'age': np.random.normal(45, 15, num_samples).astype(int),
            'support_calls': np.random.poisson(0.5, num_samples),
            'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], num_samples, p=[0.6, 0.3, 0.1]),
            'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'India'], num_samples, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
            'monthly_usage': np.random.normal(15, 5, num_samples),
            'avg_session_duration': np.random.gamma(5, 5, num_samples),
            'feature_1_usage': np.random.exponential(2, num_samples),
            'feature_2_usage': np.random.exponential(3, num_samples)
        })
        
        st.success("Sample data generated!")
        st.dataframe(sample_data)
        
        # Download sample data
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Data as CSV",
            data=csv,
            file_name="sample_customer_data.csv",
            mime="text/csv"
        )

else:  # About page
    st.header("About This Dashboard")
    st.markdown("""
    This is a proof-of-concept dashboard for customer churn prediction, developed as part of academic research.
    
    **Features:**
    - 📊 Upload your own customer data for churn prediction
    - 🎲 Generate sample data to test the system
    - 📈 Interactive results with risk categorization
    - 📥 Download predictions for further analysis
    
    **Technical Details:**
    - Built with Streamlit and Python
    - Powered by XGBoost machine learning model
    - ROC-AUC Score: 0.9960
    - Accuracy: 97.27%
    
    **Usage:**
    1. Upload a CSV file with customer data
    2. The system will preprocess and analyze the data
    3. View churn probability predictions
    4. Download results for your CRM or marketing team
    """)

# Footer
st.markdown("---")
st.markdown("*Academic Research Project - Customer Churn Prediction Proof of Concept*")