import streamlit as st
import pandas as pd
import joblib

# load models
xgboost_full = joblib.load('xgboost_full.pkl')                                          
log_reg_accessible = joblib.load('log_reg_accessible.pkl')

# load scalers
scaler_full = joblib.load('scaler_full.pkl')
scaler_accessible = joblib.load('scaler_accessible.pkl')

# load features
full_features = joblib.load('full_features.pkl')
accessible_features = joblib.load('accessible_features.pkl')

# page layout
st.set_page_config(page_title='Diabetes Risk Screening', layout='centered')
st.title('Diabetes Risk Screening Tool')
st.markdown('*Educational tool only - not a medical diagnosis.*')

# user selection: full features or accessible features
mode = st.radio('Assessment Type:', ['Self Assessment', 'Clinical Assessment'])

# user submission form
with st.form('risk_form'):
    st.subheader('Demographics & Lifestyle')
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', 20, 90, 45)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        bmi = st.number_input('BMI', 15.0, 40.0, 25.0, 0.1)
        smoking = st.selectbox('Smoking', ['No', 'Yes'])
    with col2:
        physical_activity = st.slider('Physical Activity (0-10)', 0.0, 10.0, 5.0)
        diet_quality = st.slider('Diet Quality (0-10)', 0.0, 10.0, 5.0)
        sleep_quality = st.slider('Sleep Quality (4-10)', 4.0, 10.0, 7.0)
        alcohol = st.slider('Alcohol Consumption (0-20)', 0.0, 20.0, 5.0)

    st.subheader('Medical History & Symptoms')
    col3, col4 = st.columns(2)
    with col3:
        family_hx = st.selectbox('Family History of Diabetes', ['No', 'Yes'])
        prev_pre = st.selectbox('Previous Pre-Diabetes', ['No', 'Yes'])
        hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    with col4:
        freq_urine = st.selectbox('Frequent Urination', ['No', 'Yes'])
        excess_thirst = st.selectbox('Excessive Thirst', ['No', 'Yes'])
        weight_loss = st.selectbox('Unexplained Weight Loss', ['No', 'Yes'])

    # enabling full feature entry
    if mode == 'Clinical Assessment':
        st.subheader('Lab Values')
        col5, col6 = st.columns(2)
        with col5:
            fbs = st.number_input('Fasting Blood Sugar (mg/dL)', 70.0, 200.0, 100.0)
            hba1c = st.number_input('HbA1c (%)', 4.0, 10.0, 5.5)
        with col6:
            sys_bp = st.number_input('Systolic BP', 90, 180, 120)
            dia_bp = st.number_input('Diastolic BP', 60, 120, 80)

    submitted = st.form_submit_button('Assess My Risk')

# form submission
if submitted:
    # store user input values
    user_data = {
        'Age': age,
        'Gender': 0 if gender == 'Male' else 1,
        'Ethnicity': 0,
        'SocioeconomicStatus': 1,
        'EducationLevel': 2,
        'BMI': bmi,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'AlcoholConsumption': alcohol,
        'PhysicalActivity': physical_activity,
        'DietQuality': diet_quality,
        'SleepQuality': sleep_quality,
        'FamilyHistoryDiabetes': 1 if family_hx == 'Yes' else 0,
        'GestationalDiabetes': 0,
        'PolycysticOvarySyndrome': 0,
        'PreviousPreDiabetes': 1 if prev_pre == 'Yes' else 0,
        'Hypertension': 1 if hypertension == 'Yes' else 0,
        'FrequentUrination': 1 if freq_urine == 'Yes' else 0,
        'ExcessiveThirst': 1 if excess_thirst == 'Yes' else 0,
        'UnexplainedWeightLoss': 1 if weight_loss == 'Yes' else 0,
        'BlurredVision': 0,
        'SlowHealingSores': 0,
        'TinglingHandsFeet': 0,
        'HeavyMetalsExposure': 0,
        'OccupationalExposureChemicals': 0,
        'WaterQuality': 0,
    }

    # if accessible, use logreg model
    if mode == 'Self Assessment':
        input_df = pd.DataFrame([user_data])[accessible_features]
        input_scaled = scaler_accessible.transform(input_df)
        proba = log_reg_accessible.predict_proba(input_scaled)[0, 1]
        confidence_note = 'Based on widely accessible metrics only (lower accuracy).'
    else:
        # add all other features and fill with medians if necessary
        user_data.update({
            'SystolicBP': sys_bp,
            'DiastolicBP': dia_bp,
            'FastingBloodSugar': fbs,
            'HbA1c': hba1c,
            'SerumCreatinine': 2.86,
            'BUNLevels': 28.19,
            'CholesterolTotal': 225.12,
            'CholesterolLDL': 124.92,
            'CholesterolHDL': 60.46,
            'CholesterolTriglycerides': 228.42,
            'AntihypertensiveMedications': 0,
            'Statins': 0,
            'AntidiabeticMedications': 0,
            'FatigueLevels': 4.85,
            'QualityOfLifeScore': 47.52,
            'MedicalCheckupsFrequency': 1.99,
            'MedicationAdherence': 4.84,
            'HealthLiteracy': 5.04,
        })
        # use xgboost model
        input_df = pd.DataFrame([user_data])[full_features]
        input_scaled = scaler_full.transform(input_df)
        prob = xgboost_full.predict_proba(input_scaled)[0, 1]
        confidence_note = 'Based on a full clinical assessment (higher accuracy).'

    # display results
    st.markdown('---')
    st.subheader('Your Results')

    if prob < 0.33:
        st.success(f'**Low Risk** - {prob * 100:.2f}%')
    elif prob < 0.66:
        st.warning(f'**Moderate Risk** - {prob * 100:.2f}%')
    else:
        st.error(f'**High Risk** - {prob * 100:.2f}%')

    st.progress(float(prob))
    st.caption(confidence_note)
