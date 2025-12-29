import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Logistics Risk Detection",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #ef5350;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left-color: #66bb6a;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚚 Logistics Risk Detection System</h1>', unsafe_allow_html=True)
st.markdown("### Binary Classification with Explainable AI")
st.markdown("---")

# Load model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/lightgbm_model.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        # Load feature names
        feature_names_df = pd.read_csv('data/processed/feature_names.csv')
        feature_names = feature_names_df['feature'].tolist()
        
        return model, scaler, label_encoder, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

model, scaler, label_encoder, feature_names = load_models()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/truck.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "📚 About"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Stats")
    st.metric("F1-Score", "86.23%")
    st.metric("Recall", "99.79%")
    st.metric("Precision", "75.92%")
    
    st.markdown("---")
    st.markdown("### 👨‍🎓 Author")
    st.markdown("**Mahbub Hassan**")
    st.markdown("Chulalongkorn University")
    st.markdown("[GitHub](https://github.com/mahbubchula)")

# HOME PAGE
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Samples", "32,065")
        st.caption("Jan 2021 - Aug 2024")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features", "33")
        st.caption("Temporal, Operational, Environmental")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Best Model", "LightGBM")
        st.caption("F1: 86.23%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Overview
    st.header("🎯 System Overview")
    st.markdown("""
    This system uses **Machine Learning and Explainable AI** to detect high-risk scenarios 
    in logistics operations with **99.79% recall**, ensuring minimal missed critical risks.
    
    **Key Features:**
    - ✅ Real-time risk prediction
    - ✅ SHAP-based explanations
    - ✅ 99.8% high-risk detection rate
    - ✅ Binary classification (High Risk vs Non-High Risk)
    """)
    
    # Feature Importance
    st.header("📊 Top Feature Importance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load SHAP results
        try:
            shap_df = pd.read_csv('data/results/shap_global_feature_importance.csv')
            top_10 = shap_df.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_10)))
            ax.barh(range(len(top_10)), top_10['importance'], color=colors)
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10['feature'])
            ax.set_xlabel('SHAP Importance', fontweight='bold')
            ax.set_title('Top 10 Features for Risk Detection', fontweight='bold', pad=20)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        except:
            st.info("Feature importance visualization not available")
    
    with col2:
        st.markdown("### 🔑 Key Insights")
        st.markdown("""
        **Temporal Features Dominate:**
        - Month (34%)
        - Day of Week (31%)
        - Year (27%)
        
        **These account for 92% of prediction importance!**
        
        **Operational Factors:**
        - IoT Temperature
        - Fuel Consumption
        - Historical Demand
        """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Navigate to the 'Prediction' page to try the model with your own data!")

# PREDICTION PAGE
elif page == "🔮 Prediction":
    st.header("🔮 Risk Prediction")
    
    tab1, tab2 = st.tabs(["📝 Manual Input", "📤 Upload CSV"])
    
    with tab1:
        st.markdown("### Enter Logistics Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📅 Temporal Features")
            month = st.selectbox("Month", range(1, 13), index=0)
            day_of_week = st.selectbox("Day of Week", range(7), 
                                      format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
            year = st.selectbox("Year", [2021, 2022, 2023, 2024], index=3)
            hour = st.slider("Hour of Day", 0, 23, 12)
            is_weekend = st.checkbox("Is Weekend?")
        
        with col2:
            st.subheader("🚛 Operational Features")
            fuel_consumption = st.number_input("Fuel Consumption Rate (L/100km)", 0.0, 50.0, 25.0)
            loading_time = st.number_input("Loading/Unloading Time (hours)", 0.0, 10.0, 2.0)
            historical_demand = st.number_input("Historical Demand", 0, 1000, 500)
            inventory_level = st.number_input("Warehouse Inventory Level", 0, 10000, 5000)
        
        with col3:
            st.subheader("🌦️ Environmental Features")
            temperature = st.number_input("IoT Temperature (°C)", -20.0, 50.0, 25.0)
            traffic_congestion = st.slider("Traffic Congestion Level", 0, 10, 5)
            weather_severity = st.slider("Weather Condition Severity", 0, 10, 3)
            port_congestion = st.slider("Port Congestion Level", 0, 10, 5)
        
        st.markdown("---")
        
        if st.button("🚀 Predict Risk Level", type="primary"):
            if model is not None:
                # Create feature vector (simplified - you'd need all 33 features)
                # This is a placeholder - in production, you'd collect all features
                features = np.array([[month, day_of_week, year, hour, float(is_weekend),
                                    temperature, fuel_consumption, loading_time,
                                    historical_demand, traffic_congestion, weather_severity,
                                    port_congestion, inventory_level] + [0]*20])  # Pad with zeros
                
                # Make prediction
                try:
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0]
                    
                    # Display result
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 0:  # High Risk
                            st.markdown('<div class="metric-card risk-high">', unsafe_allow_html=True)
                            st.markdown("### ⚠️ HIGH RISK")
                            st.markdown(f"**Confidence:** {probability[0]*100:.1f}%")
                            st.markdown("**Action Required:** Immediate attention needed!")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:  # Non-High Risk
                            st.markdown('<div class="metric-card risk-low">', unsafe_allow_html=True)
                            st.markdown("### ✅ NON-HIGH RISK")
                            st.markdown(f"**Confidence:** {probability[1]*100:.1f}%")
                            st.markdown("**Status:** Normal operations")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Probability chart
                        fig, ax = plt.subplots(figsize=(6, 4))
                        colors = ['#ef5350', '#66bb6a']
                        ax.bar(['High Risk', 'Non-High Risk'], probability, color=colors, alpha=0.7)
                        ax.set_ylabel('Probability')
                        ax.set_title('Risk Probability Distribution')
                        ax.set_ylim([0, 1])
                        for i, v in enumerate(probability):
                            ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')
                        st.pyplot(fig)
                    
                    st.success("✅ Prediction completed successfully!")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.error("Model not loaded. Please check model files.")
    
    with tab2:
        st.markdown("### Upload CSV File")
        st.markdown("Upload a CSV file with the following columns: " + ", ".join(feature_names[:10]) + "...")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("🚀 Predict for All Rows"):
                    # Make predictions
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)
                    
                    # Add results to dataframe
                    df['Prediction'] = ['High Risk' if p == 0 else 'Non-High Risk' for p in predictions]
                    df['High_Risk_Probability'] = probabilities[:, 0]
                    df['Confidence'] = probabilities.max(axis=1)
                    
                    # Display results
                    st.write("### Prediction Results")
                    st.dataframe(df[['Prediction', 'High_Risk_Probability', 'Confidence']])
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        high_risk_count = (predictions == 0).sum()
                        st.metric("High Risk Scenarios", high_risk_count)
                    with col2:
                        non_high_count = (predictions == 1).sum()
                        st.metric("Non-High Risk", non_high_count)
                    with col3:
                        avg_confidence = df['Confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# MODEL PERFORMANCE PAGE
elif page == "📊 Model Performance":
    st.header("📊 Model Performance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Metrics", "🎯 Confusion Matrix", "📊 Model Comparison"])
    
    with tab1:
        st.subheader("Performance Metrics (LightGBM)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("F1-Score", "86.23%", "+5.2%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recall", "99.79%", "+14.4%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", "75.92%", "+0.1%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", "75.81%", "+1.2%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Load and display figures
        st.subheader("📊 Model Comparison")
        try:
            st.image('figures/binary/01_model_comparison_binary_fixed.png', 
                    use_column_width=True)
        except:
            st.info("Model comparison figure not available")
    
    with tab2:
        st.subheader("🎯 Confusion Matrix Analysis")
        
        try:
            st.image('figures/binary/02_confusion_matrix_binary_fixed.png', 
                    use_column_width=True)
        except:
            st.info("Confusion matrix not available")
        
        st.markdown("""
        ### Key Insights:
        - ✅ **99.79% Recall**: System catches 4,855 out of 4,865 High Risk scenarios
        - ⚠️ **Trade-off**: High false positive rate (1,540 false alarms)
        - 💡 **Justification**: Better to over-caution than miss critical risks
        """)
    
    with tab3:
        st.subheader("📊 Algorithm Comparison")
        
        try:
            # Load comparison results
            results_df = pd.read_csv('data/results/model_comparison_results.csv')
            
            # Display table
            st.dataframe(
                results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
                .style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
            )
            
            # Rankings
            st.subheader("🏆 Model Rankings")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Best F1-Score:**")
                top_f1 = results_df.nlargest(3, 'F1-Score')[['Model', 'F1-Score']]
                for idx, row in top_f1.iterrows():
                    st.markdown(f"- {row['Model']}: {row['F1-Score']:.4f}")
            
            with col2:
                st.markdown("**Best Recall:**")
                top_recall = results_df.nlargest(3, 'Recall')[['Model', 'Recall']]
                for idx, row in top_recall.iterrows():
                    st.markdown(f"- {row['Model']}: {row['Recall']:.4f}")
            
            with col3:
                st.markdown("**Best Precision:**")
                top_precision = results_df.nlargest(3, 'Precision')[['Model', 'Precision']]
                for idx, row in top_precision.iterrows():
                    st.markdown(f"- {row['Model']}: {row['Precision']:.4f}")
            
        except:
            st.info("Model comparison data not available")

# ABOUT PAGE
else:
    st.header("📚 About This Project")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This system implements a **Binary Classification Framework** for detecting high-risk 
    scenarios in logistics operations using Machine Learning and Explainable AI.
    
    ### 🔬 Methodology
    
    1. **Data Preprocessing**
       - Temporal train-test split (80-20)
       - Feature engineering (33 features total)
       - StandardScaler normalization
       - SMOTE for class balancing (training only)
    
    2. **Model Selection**
       - Evaluated 7 ML algorithms
       - Selected LightGBM (best F1-score: 86.23%)
       - Prioritized recall for safety-critical deployment
    
    3. **Explainability**
       - SHAP analysis for feature importance
       - Individual prediction explanations
       - Temporal features identified as most important
    
    ### 📊 Dataset
    
    - **Size:** 32,065 samples
    - **Period:** January 2021 - August 2024
    - **Location:** Southern California logistics network
    - **Features:** 33 (temporal, operational, environmental, behavioral)
    
    ### 🎓 Author
    
    **Mahbub Hassan**  
    Transportation Engineering Researcher  
    Chulalongkorn University, Bangkok, Thailand
    
    📧 mahbub.hassan@ieee.org  
    🔗 [GitHub](https://github.com/mahbubchula/logistics-risk-detection)
    
    ### 📄 Citation
```
    @article{hassan2025logistics,
      title={Machine Learning for Critical Risk Detection in Logistics Operations},
      author={Hassan, Mahbub},
      year={2025}
    }
```
    
    ### 🙏 Acknowledgments
    
    - Dataset: Kaggle Logistics and Supply Chain Dataset
    - Institution: Chulalongkorn University
    - Frameworks: LightGBM, SHAP, scikit-learn
    
    ---
    
    **⭐ If you find this useful, please star the GitHub repository!**
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🚚 Logistics Risk Detection System | Built with Streamlit</p>
        <p>© 2025 Mahbub Hassan | Chulalongkorn University</p>
    </div>
    """,
    unsafe_allow_html=True
)
