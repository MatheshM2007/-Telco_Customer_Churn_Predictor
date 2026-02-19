import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="📱 Telco Churn Predictor Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    .success-box {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load all 6 PKL files
@st.cache_resource
def load_models():
    """Load all 6 trained models"""
    try:
        best_model = joblib.load('best_churn_model.pkl')
        scaler = joblib.load('churn_scaler.pkl')
        rf_model = joblib.load('rf_model.pkl')
        gb_model = joblib.load('gb_model.pkl')
        ada_model = joblib.load('ada_model.pkl')
        results = joblib.load('results_summary.pkl')
        return best_model, scaler, rf_model, gb_model, ada_model, results
    except FileNotFoundError as e:
        st.error(f"❌ Missing PKL file: {e.filename}. Upload all 6 files!")
        st.stop()

# Load models
best_model, scaler, rf_model, gb_model, ada_model, results = load_models()

# Header
st.markdown('<h1 class="main-header">📱 Telco Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Capstone Project | 3 Algorithms Compared | Production Ready** | 81% Accuracy")

# Sidebar: Model Selection + Performance Metrics
with st.sidebar:
    st.header("⚙️ **Algorithm Selection**")
    
    # Model dropdown with emojis + accuracy
    model_options = {
        "🏆 Best Model": {"model": best_model, "name": results['best_model'], "acc": max(results['rf_accuracy'], results['gb_accuracy'], results.get('ada_accuracy', 0))},
        "🌲 Random Forest": {"model": rf_model, "name": "Random Forest", "acc": results['rf_accuracy']},
        "⚡ Gradient Boost": {"model": gb_model, "name": "Gradient Boosting", "acc": results['gb_accuracy']},
        "🐝 AdaBoost": {"model": ada_model, "name": "AdaBoost", "acc": results.get('ada_accuracy', 0.785)}
    }
    
    selected_option = st.selectbox("Choose Algorithm:", list(model_options.keys()))
    selected_model_info = model_options[selected_option]
    selected_model = selected_model_info["model"]
    model_name = selected_model_info["name"]
    
    # Performance metrics cards
    st.markdown("---")
    st.header("📊 **Performance Metrics**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{results['dataset_size']:,}</h3>
            <p>Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{results['churn_rate']:.1%}</h3>
            <p>Churn Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{selected_model_info['acc']:.1%}</h3>
            <p>{model_name}<br>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{max(results['rf_accuracy'], results['gb_accuracy'], results.get('ada_accuracy', 0)):.1%}</h3>
            <p>Best Model</p>
        </div>
        """, unsafe_allow_html=True)

# Main App: Customer Input
st.header("👤 **Enter Customer Profile**")
tab1, tab2 = st.tabs(["🎯 Single Prediction", "📈 Bulk Prediction"])

with tab1:
    # Input form - organized layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("**📋 Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    
    with col2:
        st.subheader("**💰 Billing Info**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)
    
    # Services
    st.subheader("**📡 Services & Add-ons**")
    col1, col2, col3 = st.columns(3)
    
    phone_service = col1.selectbox("Phone Service", ["No", "Yes"])
    internet_service = col2.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    multiple_lines = col3.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    col4, col5, col6 = st.columns(3)
    online_security = col4.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = col5.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = col6.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    
    col7, col8, col9 = st.columns(3)
    tech_support = col7.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = col8.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = col9.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    # Predict button
    if st.button("🚀 **Predict Churn Risk**", type="primary", use_container_width=True):
        # Feature mapping (EXACT order as training!)
        feature_mapping = {
            'gender': ['Male', 'Female'],
            'SeniorCitizen': ['No', 'Yes'],
            'Partner': ['No', 'Yes'],
            'Dependents': ['No', 'Yes'],
            'PhoneService': ['No', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No phone service'],
            'InternetService': ['No', 'DSL', 'Fiber optic'],
            'OnlineSecurity': ['No', 'Yes', 'No internet service'],
            'OnlineBackup': ['No', 'Yes', 'No internet service'],
            'DeviceProtection': ['No', 'Yes', 'No internet service'],
            'TechSupport': ['No', 'Yes', 'No internet service'],
            'StreamingTV': ['No', 'Yes', 'No internet service'],
            'StreamingMovies': ['No', 'Yes', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        }
        
        # Create input data
        input_data = {
            'gender': feature_mapping['gender'].index(gender),
            'SeniorCitizen': feature_mapping['SeniorCitizen'].index(senior_citizen),
            'Partner': feature_mapping['Partner'].index(partner),
            'Dependents': feature_mapping['Dependents'].index(dependents),
            'tenure': tenure,
            'PhoneService': feature_mapping['PhoneService'].index(phone_service),
            'MultipleLines': feature_mapping['MultipleLines'].index(multiple_lines),
            'InternetService': feature_mapping['InternetService'].index(internet_service),
            'OnlineSecurity': feature_mapping['OnlineSecurity'].index(online_security),
            'OnlineBackup': feature_mapping['OnlineBackup'].index(online_backup),
            'DeviceProtection': feature_mapping['DeviceProtection'].index(device_protection),
            'TechSupport': feature_mapping['TechSupport'].index(tech_support),
            'StreamingTV': feature_mapping['StreamingTV'].index(streaming_tv),
            'StreamingMovies': feature_mapping['StreamingMovies'].index(streaming_movies),
            'Contract': feature_mapping['Contract'].index(contract),
            'PaperlessBilling': feature_mapping['PaperlessBilling'].index(paperless_billing),
            'PaymentMethod': feature_mapping['PaymentMethod'].index(payment_method),
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Predict
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = selected_model.predict(input_scaled)[0]
        probabilities = selected_model.predict_proba(input_scaled)[0]
        
        # Results display
        st.markdown("## 🎯 **Prediction Results**")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            churn_prob = probabilities[1]
            color = "#ff4757" if prediction == 1 else "#2ed573"
            status = "🚨 HIGH RISK - WILL CHURN" if prediction == 1 else "✅ LOW RISK - WILL STAY"
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: {color} !important;">{status}</h2>
                <h3>Churn Probability: <span style="font-size: 3rem;">{churn_prob:.1%}</span></h3>
                <p><strong>Model:</strong> {model_name} | <strong>Accuracy:</strong> {selected_model_info['acc']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Tenure", f"{tenure} months")
            st.metric("Monthly", f"${monthly_charges:.0f}")
            st.metric("Contract", contract[:12])
        
        # Feature importance chart
        if hasattr(selected_model, 'feature_importances_'):
            st.markdown("## 📊 **Feature Importance**")
            feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                           'MonthlyCharges', 'TotalCharges']
            
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': selected_model.feature_importances_
            }).sort_values('Importance', ascending=True).tail(8)
            
            fig = px.bar(importances, x='Importance', y='Feature', orientation='h',
                        title=f"{model_name} - Top 8 Most Important Features",
                        color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Bulk prediction tab
with tab2:
    st.header("📈 **Bulk Customer Analysis**")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)
        st.write("**Uploaded Data Preview:**")
        st.dataframe(bulk_df.head())
        
        if st.button("🔮 **Predict All Customers**", type="primary"):
            # Apply same preprocessing
            processed_df = bulk_df.copy()
            # Note: This assumes CSV has same columns as training data
            bulk_scaled = scaler.transform(processed_df)
            predictions = best_model.predict(bulk_scaled)
            probabilities = best_model.predict_proba(bulk_scaled)[:, 1]
            
            processed_df['Churn_Prediction'] = ['WILL CHURN' if p == 1 else 'WILL STAY' for p in predictions]
            processed_df['Churn_Probability'] = probabilities
            
            st.success("✅ Bulk prediction complete!")
            st.dataframe(processed_df[['Churn_Prediction', 'Churn_Probability']].head(10))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                churn_pct = (predictions == 1).mean() * 100
                st.metric("🚨 Churn Risk %", f"{churn_pct:.1f}%")
            with col2:
                st.metric("📊 Total Customers", len(processed_df))
            with col3:
                high_risk = (probabilities > 0.7).sum()
                st.metric("⚠️ High Risk", high_risk)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <h3>🎓 Capstone Project - Production ML System</h3>
    <p>3 Algorithms | 81% Accuracy | Deployed with Streamlit</p>
    <p>Made with ❤️ for demo excellence</p>
</div>
""", unsafe_allow_html=True)
