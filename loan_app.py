import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Loan Approval Predictor", layout="wide", initial_sidebar_state="expanded")

st.title("🏦 Loan Approval Prediction System")

# Get the directory where the app is running
app_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

# Load models and encoders with better error handling
@st.cache_resource
def load_models():
    """Load all trained models and supporting files"""
    try:
        model_files = {
            'best_model.pkl': 'best_model.pkl',
            'all_models.pkl': 'all_models.pkl',
            'label_encoders.pkl': 'label_encoders.pkl',
            'feature_names.pkl': 'feature_names.pkl',
            'results.pkl': 'results.pkl'
        }
        
        data = {}
        for key, filename in model_files.items():
            # Try multiple paths
            paths_to_try = [
                Path(filename),
                app_dir / filename,
                Path.cwd() / filename,
                Path(__file__).parent / filename if '__file__' in globals() else None
            ]
            
            file_path = None
            for path in paths_to_try:
                if path and path.exists():
                    file_path = path
                    break
            
            if file_path is None:
                raise FileNotFoundError(f"Cannot find {filename} in any of the expected locations")
            
            with open(file_path, 'rb') as f:
                data[key] = pickle.load(f)
        
        return (data['best_model.pkl'], data['all_models.pkl'], 
                data['label_encoders.pkl'], data['feature_names.pkl'], 
                data['results.pkl'])
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("📝 Note: Ensure all .pkl files are in the same directory as this app")
        return None, None, None, None, None

# Load the models
best_model, all_models, label_encoders, feature_names, results = load_models()
models_loaded = all([best_model, all_models, label_encoders, feature_names, results])

if not models_loaded:
    st.error("⚠️ Models could not be loaded. Please ensure all model files are available.")
    st.stop()

# Display model performance
with st.expander("📊 Model Performance Summary", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy Comparison")
        accuracy_data = {name: results[name]['accuracy'] for name in results}
        accuracy_df = pd.DataFrame(list(accuracy_data.items()), columns=['Model', 'Accuracy'])
        accuracy_df = accuracy_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        st.dataframe(accuracy_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Best Model Details")
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_results = results[best_model_name]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Model", best_model_name)
            st.metric("Accuracy", f"{best_results['accuracy']:.2%}")
        with col_b:
            st.metric("Precision", f"{best_results['precision']:.2%}")
            st.metric("Recall", f"{best_results['recall']:.2%}")

st.markdown("---")

# Input section
st.header("📝 Enter Loan Application Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=1500, step=100)

with col2:
    loan_amount = st.number_input("Loan Amount ($000)", min_value=0, value=100, step=10)
    loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=480, value=360, step=12)
    employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=5, step=1)

with col3:
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0, step=1)
    credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=10)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])

with col2:
    married = st.selectbox("Married", ['Yes', 'No'])

with col3:
    dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])

with col4:
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])

col1, col2, col3, col4 = st.columns(4)

with col1:
    self_employed = st.selectbox("Self Employed", ['Yes', 'No'])

with col2:
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

with col3:
    home_ownership = st.selectbox("Home Ownership", ['Own', 'Mortgage', 'Rent'])

with col4:
    loan_purpose = st.selectbox("Loan Purpose", 
                                 ['Home', 'Auto', 'Education', 'Personal', 'Business', 'Debt Consolidation', 'Car'])

st.markdown("---")

# Make prediction
if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True, key="predict_btn"):
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self Employed': [self_employed],
            'Property Area': [property_area],
            'Home Ownership': [home_ownership],
            'Loan Purpose': [loan_purpose],
            'Age': [age],
            'Applicant Income($)': [applicant_income],
            'Coapplicant Income($)': [coapplicant_income],
            'Loan Amount($000)': [loan_amount],
            'Loan Term(months)': [loan_term],
            'Credit History': [credit_history],
            'Credit Score': [credit_score],
            'Employment Years': [employment_years],
            'Existing Loans': [existing_loans]
        })
        
        # Encode categorical variables
        for col in input_data.columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Ensure feature order matches training data
        input_data = input_data[feature_names]
        
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Make predictions with all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in all_models.items():
            pred = model.predict(input_data)[0]
            
            # Get probability for models that support it
            try:
                prob = model.predict_proba(input_data)[0]
                probabilities[model_name] = prob[1]  # Probability of approval (class 1)
            except:
                probabilities[model_name] = None
            
            predictions[model_name] = pred
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("All Model Predictions")
            predictions_df = pd.DataFrame({
                'Model': list(predictions.keys()),
                'Prediction': ['✅ APPROVED' if p == 1 else '❌ REJECTED' for p in predictions.values()],
                'Approval Probability': [f"{probabilities[m]*100:.2f}%" if probabilities[m] is not None else "N/A" 
                                       for m in predictions.keys()]
            })
            st.dataframe(predictions_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Best Model Prediction")
            best_model_name = max(results, key=lambda x: results[x]['accuracy'])
            best_prediction = predictions[best_model_name]
            best_prob = probabilities[best_model_name]
            
            if best_prediction == 1:
                st.success("### ✅ LOAN APPROVED", icon="✅")
                st.metric("Approval Probability", f"{best_prob*100:.2f}%")
            else:
                st.error("### ❌ LOAN REJECTED", icon="❌")
                st.metric("Rejection Probability", f"{(1-best_prob)*100:.2f}%")
            
            st.info(f"**Model:** {best_model_name}\n**Accuracy:** {results[best_model_name]['accuracy']:.2%}")
        
        # Summary
        st.markdown("---")
        approval_count = sum(1 for p in predictions.values() if p == 1)
        total_models = len(predictions)
        
        st.success(f"✅ Summary: {approval_count} out of {total_models} models predict approval", icon="ℹ️")
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("Please check your inputs and try again")

st.markdown("---")
st.caption("🤖 ML-based Loan Approval Prediction System | Trained on 10,000 loan applications | Best Model: Gradient Boosting (91.85% accuracy)")
