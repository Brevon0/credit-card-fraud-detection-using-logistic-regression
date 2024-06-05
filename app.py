import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import hashlib
import pickle

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Save the trained model
pickle.dump(model, open('model.pkl', 'wb'))

# Create Streamlit app
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Verify user credentials
def verify_credentials(username, password):
    # Hash the input password
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    # Check if the username and hashed input password match the stored values
    if username == "admin" and password_hash == "4126911a3b6c2506eae2ba766ba68776ecbb2fc34c36244b26ae12cdeebe3d5d":
        return True
    else:
        return False

# Show the login page
def show_login_page():
    st.title("Credit Card Fraud Detection Model")
    st.write("Please enter your credentials to access the app.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if verify_credentials(username, password):
            # Set login state to True
            st.session_state.login_state = True
        else:
            st.error("Invalid username or password.")

# Show the app
def show_app():
    st.title("Credit Card Fraud Detection Model")
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    # Load the trained model
    model = pickle.load(open('model.pkl', 'rb'))

    # Create input fields for user to enter feature values
    input_df = st.text_input('Input All features', placeholder='Enter features separated by commas')
    input_df_lst = input_df.split(',')

    # Create a button to submit input and get prediction
    submit = st.button("Submit")

    if submit:
        # Get input feature values
        features = np.array(input_df_lst, dtype=np.float64)

        # Make prediction
        prediction = model.predict(features.reshape(1, -1))

        # Display result
        if prediction[0] == 0:
            st.success("Legitimate transaction")
        else:
            st.error("Fraudulent transaction")

# Start the app and show the login page
if "login_state" not in st.session_state:
    st.session_state.login_state = False

if st.session_state.login_state:
    show_app()
else:
    show_login_page()
