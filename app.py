import streamlit as st
import pandas as pd
import joblib 
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load('machine_model.pkl')
kmeans = model['kmeans_model']
pca = model['pca']
scaler = model['scaler']
expected_features = [
    'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Estimated Savings (k$)', 
    'Credit Score', 'Loyalty Years', 'Gender_Female', 'Gender_Male', 
    'Preferred Category_Budget', 'Preferred Category_Electronics', 
    'Preferred Category_Fashion', 'Preferred Category_Luxury',
    'Age Group_18-25', 'Age Group_26-35', 'Age Group_36-50', 'Age Group_51-65', 'Age Group_65+'
]


cluster_descriptions = {
    0: "These customers represent a large number of the customer base and prefer electronics. They are loyal and financially stable, however their spending is not as high, and not a priority. They could be targetted when doing loyalty deals, as well as offers on expensive electronics.",
    1: "These customers have very high income, however they are not spending their money at this shop, they also prefer electronics. They are a prime target for high-end marketing campaigns, that include exclusive products which are intended to make them spend more.",
    2: "These customers are young, with a low income however spend a lot at the store, primarily shopping luxury items. This group is highly infleunced by trends and they are willing to spend a signficant portion of their income. They could be targeted for offers specific to new products and limited-time products.",
    3: "These are the store's most valuable customers as they have a high income and also spend a lot at the shop, again primarily focusing on luxury items. Marketing here should focus on special services and various offers to maintain their loyalty and high spending habits.",
    4: "These customers are young with a decent income, however they are careful when spending. They prefer fashion items. Marketing efforts should be made to encourage more spending through offers and promotions as there is a lot of opportunity here."
}

cluster_groups = {
    0: "Established customer",
    1: "High-earning customer",
    2: "Young high-spender",
    3: "Fashionable spender",
    4: "Budget spenders"
}

st.set_page_config(page_title="Customer Grouper Dashboard", layout="wide", page_icon="👥")

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎯 Customer Segmentation Dashboard</h1>
    <p>AI-powered customer grouping using K-Means clustering</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.header("Predict a New Customer's Group")
st.write("Enter the details of a new customer below to see which group they belong to.")

col1, spacer, col2 = st.columns([1, 0.2, 1])

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
    gender_options = ['Female', 'Male']
    gender = st.radio("Gender", gender_options, horizontal=True)
    
    st.subheader("Financial Information")
    annual_income = st.number_input("Annual Income (k$)", min_value=10, max_value=137, value=50, step=1)
    estimated_savings = st.number_input("Estimated Savings (k$)", min_value=2.0, max_value=125.0, value=25.0, step=0.5)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600, step=10)

with col2:
    st.subheader("Shopping Behavior")
    spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=99, value=50, step=1)
    preferred_category_options = ['Budget', 'Electronics', 'Fashion', 'Luxury']
    preferred_category = st.selectbox("Preferred Category", preferred_category_options)
    
    st.subheader("Loyalty Information")
    loyalty_years = st.number_input("Loyalty Years", min_value=0, max_value=10, value=3, step=1)

def get_age_group(age):
    if 18 <= age <= 25:
        return '18-25'
    elif 26 <= age <= 35:
        return '26-35'
    elif 36 <= age <= 50:
        return '36-50'
    elif 51 <= age <= 65:
        return '51-65'
    else:
        return '65+'

age_group = get_age_group(age)

st.markdown("---")
predict_button = st.button("Predict Customer Group", type="primary", use_container_width=True)

if predict_button:
    # Create a DataFrame for the new customer with all required columns
    new_customer_df = pd.DataFrame({
        'Age': [age], 
        'Annual Income (k$)': [annual_income],
        'Spending Score (1-100)': [spending_score],
        'Estimated Savings (k$)': [estimated_savings],
        'Credit Score': [credit_score],
        'Loyalty Years': [loyalty_years],
        'Gender_Female': [1 if gender == 'Female' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'Preferred Category_Budget': [1 if preferred_category == 'Budget' else 0],
        'Preferred Category_Electronics': [1 if preferred_category == 'Electronics' else 0],
        'Preferred Category_Fashion': [1 if preferred_category == 'Fashion' else 0],
        'Preferred Category_Luxury': [1 if preferred_category == 'Luxury' else 0],
        'Age Group_18-25': [1 if age_group == '18-25' else 0],
        'Age Group_26-35': [1 if age_group == '26-35' else 0],
        'Age Group_36-50': [1 if age_group == '36-50' else 0],
        'Age Group_51-65': [1 if age_group == '51-65' else 0],
        'Age Group_65+': [1 if age_group == '65+' else 0]
    })
    
    new_customer_df = new_customer_df[expected_features] # Reorder the columns to match the order used in training
    
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Estimated Savings (k$)', 'Credit Score', 'Loyalty Years']
    categorical_cols = [col for col in expected_features if col not in numerical_cols]

    numerical_data = new_customer_df[numerical_cols]
    categorical_data = new_customer_df[categorical_cols]
    
    # Scale numerical features and combine with categorical features
    numerical_scaled = scaler.transform(numerical_data)   
    new_customer_scaled = np.concatenate([numerical_scaled, categorical_data.values], axis=1)   
    
    predicted_cluster = kmeans.predict(new_customer_scaled)[0]
    
    st.subheader(f"This customer belongs to **Cluster {predicted_cluster}** ({cluster_groups[predicted_cluster]})")
    st.write(cluster_descriptions[predicted_cluster])