import streamlit as st
import pandas as pd
import joblib

# Load model artifacts
xgboost_full = joblib.load('xgboost_full.pkl')
log_reg_accessible = joblib.load('log_reg_accessible.pkl')
scaler_full = joblib.load('scaler_full.pkl')
scaler_accessible = joblib.load('scaler_accessible.pkl')
full_features = joblib.load('full_features.pkl')
accessible_features = joblib.load('accessible_features.pkl')

# Page configuration
st.set_page_config(page_title='Diabetes Risk Screening', layout='wide')
st.title('Diabetes Risk Screening Tool')
st.markdown('*Educational tool only - not a medical diagnosis.*')

# Assessment mode selection
mode = st.radio(
    'Assessment Type:',
    ['Quick Screening (self-reported only)', 'Full Assessment (includes lab values)']
)

def append_digit(field, digit):
    current = st.session_state[field]
    # Replace default placeholder on first click
    if field == 'age_value' and current == '45':
        st.session_state[field] = digit
    elif field == 'bmi_value' and current == '25.0':
        st.session_state[field] = digit
    else:
        st.session_state[field] = current + digit

def clear_field(field):
    st.session_state[field] = ''

def backspace_field(field):
    st.session_state[field] = st.session_state[field][:-1]

def numpad(field_name, label):
    st.markdown(f"**{label}**")
    st.text(f"Current value: {st.session_state[field_name] or '(empty)'}")
    
    # Digit buttons arranged in a 3x3 grid + bottom row
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)
    
    digits_grid = [
        (row1, ['1', '2', '3']),
        (row2, ['4', '5', '6']),
        (row3, ['7', '8', '9']),
    ]
    
    for row, digits in digits_grid:
        for col, digit in zip(row, digits):
            col.button(digit, key=f'{field_name}_{digit}',
                       on_click=append_digit, args=(field_name, digit),
                       use_container_width=True)
    
    # Bottom row: decimal (BMI only), 0, backspace
    if field_name == 'bmi_value':
        row4[0].button('.', key=f'{field_name}_dot',
                       on_click=append_digit, args=(field_name, '.'),
                       use_container_width=True)
    else:
        row4[0].button('C', key=f'{field_name}_clear',
                       on_click=clear_field, args=(field_name,),
                       use_container_width=True)
    
    row4[1].button('0', key=f'{field_name}_0',
                   on_click=append_digit, args=(field_name, '0'),
                   use_container_width=True)
    row4[2].button('⌫', key=f'{field_name}_back',
                   on_click=backspace_field, args=(field_name,),
                   use_container_width=True)

# Input form
with st.form('risk_form'):
    st.subheader('Demographics & Lifestyle')
    col1, col2 = st.columns(2)
    with col1:
        numpad('age_value', 'Age')
        try:
            age = int(st.session_state.age_value) if st.session_state.age_value else 45
        except ValueError:
            age = 45
        gender = st.selectbox('Gender', ['Male', 'Female'])
        numpad('bmi_value', 'BMI')
        try:
            bmi = float(st.session_state.bmi_value) if st.session_state.bmi_value else 25.0
        except ValueError:
            bmi = 25.0
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

    # Lab values shown only in Full Assessment mode
    if mode == 'Full Assessment (includes lab values)':
        st.subheader('Lab Values')
        col5, col6 = st.columns(2)
        with col5:
            fbs = st.number_input('Fasting Blood Sugar (mg/dL)', 70.0, 200.0, 100.0)
            hba1c = st.number_input('HbA1c (%)', 4.0, 10.0, 5.5)
        with col6:
            sys_bp = st.number_input('Systolic BP', 90, 180, 120)
            dia_bp = st.number_input('Diastolic BP', 60, 120, 80)

    submitted = st.form_submit_button('Assess My Risk')

# Process submission
if submitted:
    # Build user input dictionary with accessible features
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

    if mode == 'Quick Screening (self-reported only)':
        input_df = pd.DataFrame([user_data])[accessible_features]
        input_scaled = scaler_accessible.transform(input_df)
        proba = log_reg_accessible.predict_proba(input_scaled)[0, 1]
        confidence_note = 'Based on accessible features only (lower accuracy).'
    else:
        # Add clinical values and remaining features using population medians
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
        input_df = pd.DataFrame([user_data])[full_features]
        input_scaled = scaler_full.transform(input_df)
        proba = xgboost_full.predict_proba(input_scaled)[0, 1]
        confidence_note = 'Based on full clinical assessment (high accuracy).'

    # Display results
    st.markdown('---')
    st.subheader('Your Results')

    if proba < 0.3:
        st.success(f'**Low Risk** - {proba * 100:.1f}%')
    elif proba < 0.6:
        st.warning(f'**Moderate Risk** - {proba * 100:.1f}%')
    else:
        st.error(f'**High Risk** - {proba * 100:.1f}%')

    st.progress(float(proba))
    st.caption(confidence_note)

    st.info(
        'This tool is for educational purposes only and does not constitute a medical '
        'diagnosis. If you have concerns about your risk, consult a healthcare provider.'
    )
