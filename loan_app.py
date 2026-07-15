import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# Load models and encoders
@st.cache_resource
def load_models():
    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    
    with open('all_models.pkl', 'rb') as f:
        all_models = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    return best_model, all_models, label_encoders, feature_names, results

try:
    best_model, all_models, label_encoders, feature_names, results = load_models()
    models_loaded = True
except:
    models_loaded = False
    st.error("⚠️ Models not found! Please run model_training.py first.")

# Title and description
st.title("🏦 Loan Approval Prediction System")
st.markdown("---")

if models_loaded:
    # Display model performance
    with st.expander("📊 Model Performance Summary"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Model Accuracy Comparison")
            accuracy_data = {name: results[name]['accuracy'] for name in results}
            accuracy_df = pd.DataFrame(list(accuracy_data.items()), columns=['Model', 'Accuracy'])
            accuracy_df = accuracy_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
            
            st.dataframe(accuracy_df, use_container_width=True)
        
        with col2:
            st.write("### Best Model Performance")
            best_model_name = max(results, key=lambda x: results[x]['accuracy'])
            best_results = results[best_model_name]
            
            metrics = {
                'Model': best_model_name,
                'Accuracy': f"{best_results['accuracy']:.4f}",
                'Precision': f"{best_results['precision']:.4f}",
                'Recall': f"{best_results['recall']:.4f}",
                'F1-Score': f"{best_results['f1']:.4f}"
            }
            
            for key, value in metrics.items():
                st.metric(label=key, value=value)
    
    st.markdown("---")
    
    # Input section
    st.write("## 📝 Enter Loan Application Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=1500, step=100)
    
    with col2:
        loan_amount = st.number_input("Loan Amount ($000)", min_value=0, value=100, step=10)
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=480, value=360, step=12)
        employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
    
    with col3:
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
        credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    
    st.write("---")
    
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
    if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
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
            st.write("## 🎯 Prediction Results")
            
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
                st.write("### All Model Predictions")
                predictions_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Prediction': ['✅ APPROVED' if p == 1 else '❌ REJECTED' for p in predictions.values()],
                    'Approval Probability': [f"{probabilities[m]*100:.2f}%" if probabilities[m] is not None else "N/A" 
                                           for m in predictions.keys()]
                })
                st.dataframe(predictions_df, use_container_width=True)
            
            with col2:
                st.write("### Best Model Prediction")
                best_model_name = max(results, key=lambda x: results[x]['accuracy'])
                best_prediction = predictions[best_model_name]
                best_prob = probabilities[best_model_name]
                
                if best_prediction == 1:
                    st.success(f"### ✅ LOAN APPROVED")
                    st.metric("Approval Probability", f"{best_prob*100:.2f}%")
                else:
                    st.error(f"### ❌ LOAN REJECTED")
                    st.metric("Rejection Probability", f"{(1-best_prob)*100:.2f}%")
                
                st.write(f"*Model: {best_model_name}*")
                st.write(f"*Accuracy: {results[best_model_name]['accuracy']:.4f}*")
            
            # Summary
            st.markdown("---")
            approval_count = sum(1 for p in predictions.values() if p == 1)
            total_models = len(predictions)
            
            st.write(f"### Summary: {approval_count} out of {total_models} models predict approval")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    st.markdown("---")
    st.write("*Model trained on loan application dataset with multiple classification algorithms*")

else:
    st.info("Please run `model_training.py` to train the models first.")
