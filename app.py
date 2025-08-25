import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

# Load the trained model
with open('customer.pickle', 'rb') as file:
    final_model = pickle.load(file)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .sidebar .sidebar-content { background-color: #fafafa; }
    .title { color: #2c3e50; text-align: center; font-size: 38px; font-weight: bold; }
    .subtitle { color: #16a085; text-align: center; font-size: 20px; margin-bottom: 20px; }
    .footer { text-align: center; color: gray; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<div class='title'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a customer will churn using ML</div>", unsafe_allow_html=True)

# Tabs for Navigation
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction", "📊 Data Insights", "ℹ️ About"])

# ---------------------------
# SINGLE PREDICTION
# ---------------------------
with tab1:
    st.sidebar.header("🔍 Input Features")
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    senior_citizen = st.sidebar.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Partner", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    dependents = st.sidebar.selectbox("Dependents", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    tenure = st.sidebar.slider("Tenure (months)", min_value=1, max_value=72, value=1)
    phone_service = st.sidebar.selectbox("Phone Service", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    multiple_lines = st.sidebar.selectbox("Multiple Lines", options=[0, 1, 2], format_func=lambda x: "Yes" if x == 1 else ("No" if x == 0 else "No phone service"))
    internet_service = st.sidebar.selectbox("Internet Service", options=[0, 1, 2], format_func=lambda x: "DSL" if x == 1 else ("Fiber Optic" if x == 2 else "No"))
    online_security = st.sidebar.selectbox("Online Security", options=[0, 1, 2], format_func=lambda x: "Yes" if x == 1 else ("No" if x == 0 else "No internet service"))
    online_backup = st.sidebar.selectbox("Online Backup", options=[0, 1, 2], format_func=lambda x: "Yes" if x == 1 else ("No" if x == 0 else "No internet service"))
    device_protection = st.sidebar.selectbox("Device Protection", options=[0, 1, 2], format_func=lambda x: "Yes" if x == 1 else ("No" if x == 0 else "No internet service"))
    tech_support = st.sidebar.selectbox("Tech Support", options=[0, 1, 2], format_func=lambda x: "Yes" if x == 1 else ("No" if x == 0 else "No internet service"))
    streaming_tv = st.sidebar.selectbox("Streaming TV", options=[0, 1, 2], format_func=lambda x: "Yes" if x == 1 else ("No" if x == 0 else "No internet service"))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", options=[0, 1, 2], format_func=lambda x: "Yes" if x == 1 else ("No" if x == 0 else "No internet service"))
    contract = st.sidebar.selectbox("Contract", options=[0, 1, 2], format_func=lambda x: "Month-to-Month" if x == 0 else ("One Year" if x == 1 else "Two Year"))
    paperless_billing = st.sidebar.selectbox("Paperless Billing", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    payment_method = st.sidebar.selectbox("Payment Method", options=[0, 1, 2, 3], format_func=lambda x: ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"][x])
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=0.0, step=0.01)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=0.0, step=0.01)

    st.subheader("Prediction Result")
    if st.button("Predict Churn"):
        input_data = np.array([[gender, senior_citizen, partner, dependents, tenure, phone_service,
                                 multiple_lines, internet_service, online_security, online_backup,
                                 device_protection, tech_support, streaming_tv, streaming_movies,
                                 contract, paperless_billing, payment_method, monthly_charges, total_charges]])

        prediction = final_model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ The customer is likely to **Churn**.")
        else:
            st.success("✅ The customer is **Not likely to Churn**.")

# ---------------------------
# BATCH PREDICTION
# ---------------------------
with tab2:
    st.subheader("📂 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", df.head())

        if st.button("Run Batch Prediction"):
            predictions = final_model.predict(df)
            df["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]

            st.success("✅ Batch Prediction Completed!")
            st.write(df)

            # Download link for result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Predictions as CSV",
                               data=csv,
                               file_name="batch_predictions.csv",
                               mime="text/csv")

# ---------------------------
# DATA INSIGHTS
# ---------------------------
with tab3:
    st.subheader("Visual Insights")
    churn_data = {"Churn": ["Yes", "No"], "Count": [400, 600]}  # Example data
    fig = px.pie(churn_data, names="Churn", values="Count", title="Churn Distribution",
                 color="Churn", color_discrete_map={"Yes": "red", "No": "green"})
    st.plotly_chart(fig)

# ---------------------------
# ABOUT
# ---------------------------
with tab4:
    st.subheader("About This Project")
    st.write("""
    This **Customer Churn Prediction App** predicts whether a customer will leave the service based on their features.

    **About Developer:**
    - 👩‍💻 **Name:** Sidra Hussain  
    - 🎓 **B.Tech CSE Student** at Graphic Era Deemed to be University  
    - 🔍 **Focus:** AI & Machine Learning  
    - 📫 **Email:** hsidra10@gmail.com
    """)
    st.write("**Connect with me:**")
    st.markdown("""
    <a href='https://github.com/sidrah-star' target='_blank'><img src='https://cdn-icons-png.flaticon.com/512/733/733553.png' width='30'></a>
    &nbsp;&nbsp;
    <a href='https://www.linkedin.com/in/sidra-hussain123/' target='_blank'><img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='30'></a>
    """, unsafe_allow_html=True)

st.markdown("<hr class='footer'>Made with ❤️ by Sidra Hussain</hr>", unsafe_allow_html=True)
