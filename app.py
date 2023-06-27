import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the model
model = pickle.load(open('best_model.pkl', 'rb'))

features = ['satisfaction_level', 'last_evaluation', 'number_project',
            'average_montly_hours', 'time_spend_company', 'Work_accident',
            'promotion_last_5years', 'Department', 'salary']

# Encoding Categorical Variables
def cat_encode(df):
    le = LabelEncoder()
    df['Work_accident'] = df['Work_accident'].map({'Yes': 1, 'No': 0})
    df['promotion_last_5years'] = df['promotion_last_5years'].map({'Yes': 1, 'No': 0})
    df['Department'] = le.fit_transform(df['Department'])
    df['salary'] = le.fit_transform(df['salary'])
    return df

# Create a function to get user input
def get_user_input():
    # You can create as many input fields as you need
    # For example, if you have a feature 'satisfaction_level', you can get it like this:
    satisfaction_level = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
    last_evaluation = st.sidebar.slider('Last Evaluation Score', 0.0, 1.0, 0.5)
    number_project = st.sidebar.number_input('Total Projects Done', 0, 15)
    average_montly_hours = st.sidebar.slider('Average Monthly hours worked', 0, 240, 720)
    time_spend_company = st.sidebar.slider('Time spent at Company (Years)', 0, 5, 30)
    Work_accident = st.sidebar.radio('Work Accident', ['Yes', 'No'])
    promotion_last_5years = st.sidebar.radio('Promotion in Last 5 Years', ['Yes', 'No'])
    Department = st.sidebar.radio('Department', ['sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD'])
    salary = st.sidebar.radio('Salary Category', ['low', 'medium', 'high'])


    # Collect all inputs into a dictionary
    user_input = {'satisfaction_level': satisfaction_level,
                  'last_evaluation': last_evaluation,
                  'number_project': number_project,
                  'average_montly_hours': average_montly_hours,
                  'time_spend_company': time_spend_company,
                  'Work_accident': Work_accident,
                  'promotion_last_5years': promotion_last_5years,
                  'Department': Department,
                  'salary': salary
                  }
    # Transform the data into a data frame
    data = pd.DataFrame(user_input, index=[0])

    # Encode Categorical Variables to Numerical
    enc_data = cat_encode(data)

    # Return Preprocessed data
    return enc_data

# Main function to define the structure of the web app
def main():
    st.write("# Employee Churn Prediction App")
    st.write("Enter the details of the employee to predict if they will leave the company.")
    user_input = get_user_input()
    prediction = model.predict(user_input)
    if prediction[0] == 1:
        st.write("The employee is likely to leave the company.")
    else:
        st.write("The employee is likely to stay in the company.")

if __name__ == '__main__':
    main()
