import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the model
model = pickle.load(open('best_model.pkl', 'rb'))

# Create a function to get user input
def get_user_input():
    # You can create as many input fields as you need
    # For example, if you have a feature 'satisfaction_level', you can get it like this:
    satisfaction_level = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
    # Collect all inputs into a dictionary
    user_input = {'satisfaction_level': satisfaction_level,
                  # Add other inputs here
                  }
    # Transform the data into a data frame
    data = pd.DataFrame(user_input, index=[0])
    return data

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
