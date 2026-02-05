import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Churn Prediction App",
    layout="centered"
)

st.title("Churn Prediction")
st.write("Predict churn rate prediction using ml models")

model=joblib.load("best_churn_pipeline.pkl")
uniques=joblib.load("categorical_values.pkl")

st.subheader("customer details")

col1,col2,col3=st.columns(3)

with col1:
    gender=st.selectbox("Gender",uniques["gender"])
    SeniorCitizen = st.checkbox("Senior Citizen")
    Partner = st.checkbox("Has Partner")
    Dependents = st.checkbox("Has Dependents")
    PhoneService = st.checkbox("Phone Service")
    PaperlessBilling = st.checkbox("Paperless Billing")  
    

with col2:
    InternetService=st.selectbox("Internet Service",uniques["InternetService"])
    MultipleLines=st.selectbox("Multiple Lines",uniques["MultipleLines"])
    OnlineSecurity=st.selectbox("Online Security",uniques["OnlineSecurity"])
    OnlineBackup=st.selectbox("Online Backup",uniques["OnlineBackup"])
    DeviceProtection=st.selectbox("Device Service",uniques["DeviceProtection"])
    TechSupport=st.selectbox("Tech Support",uniques["TechSupport"])

with col3:
    StreamingTV=st.selectbox("Streaming TV",uniques["StreamingTV"])
    StreamingMovies=st.selectbox("Streaming Movies",uniques["StreamingMovies"])
    Contract=st.selectbox("Contract",uniques["Contract"])
    PaymentMethod=st.selectbox("Payment Method",uniques["PaymentMethod"])
    tenure=st.slider("tenure",0,71,12)
    MonthlyCharges=st.slider("Montlhy Charges",18.0,120.0,70.0)

TotalCharges=tenure*MonthlyCharges

num_services=sum([
    PhoneService,
    MultipleLines=="Yes",
    OnlineSecurity=="Yes",
    OnlineBackup=="Yes",
    DeviceProtection=="Yes",
    TechSupport=="Yes",
    StreamingTV=="Yes",
    StreamingMovies=="Yes"
])

input_df=pd.DataFrame([{
    'gender':gender,
    'SeniorCitizen':1 if SeniorCitizen else 0,
    'Partner':Partner,
    'Dependents':Dependents,
    'tenure':tenure,
    'PhoneService':PhoneService,
    'MultipleLines':MultipleLines,
    'InternetService':InternetService,
    'OnlineSecurity':OnlineSecurity,
    'OnlineBackup':OnlineBackup,
    'DeviceProtection':DeviceProtection,
    'TechSupport':TechSupport,
    'StreamingTV':StreamingTV,
    'StreamingMovies':StreamingMovies,
    'Contract':Contract,
    'PaperlessBilling':PaperlessBilling,
    'PaymentMethod':PaymentMethod,
    'MonthlyCharges':MonthlyCharges,
    'num_services':num_services,
    'TotalCharges':TotalCharges
}])

st.divider()

if st.button("Predict Churn",use_container_width=True):
    churn_prob=model.predict_proba(input_df)[0,1]
    decision="YES" if churn_prob>0.5 else "No"

    st.subheader("Prediction Result")
    st.metric(label="Churn probability", value= f"{churn_prob:.2%}")

    if churn_prob>=0.5:
        st.error("Decision: Customer likely to churn")
        st.write("High churn risk. Retention action recommended.")
    else:
        st.success("Decision: Customer unlikely to churn")
        st.write("Low churn risk. Customer likely to stay.")
