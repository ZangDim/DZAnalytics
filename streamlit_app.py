import bcrypt
import streamlit as st
from time import sleep
from navigation import make_sidebar

# Set custom page configuration
st.set_page_config(
    page_title="Sakila Analytics",  # Custom window title
    page_icon="🎬",  # Optional: Set a custom icon (emoji or image URL)
    layout="centered"  # Optional: Choose between "centered" or "wide" layout
)

make_sidebar()

# Function to retrieve stored hashed password for a given username
def get_stored_password_hash(username):
    with open("passwords.txt", "r") as f:
        for line in f:
            stored_username, stored_hash = line.strip().split(":")
            if stored_username == username:
                return stored_hash
    return None  # Return None if the username does not exist

# Streamlit app logic for the login page
st.title("Sakila Analytics - Login Page 🔒")

st.write("This app was created as a demonstration and is part of `Dimitris Zanganas's` portfolio.")
st.write("Please log in to continue (username `admin`, password `admin`).")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Log in", type="primary"):
    stored_hash = get_stored_password_hash(username)
    
    if stored_hash:
        # Compare entered password with stored hash
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            st.session_state.logged_in = True
            st.session_state.username = username  # Store the username in session state
            st.success("Logged in successfully!")
            sleep(0.5)
            st.write(f"Welcome, {username}!")  # Display the username upon login
            st.switch_page("pages/home.py")  # Redirect to home page
        else:
            st.error("Incorrect username or password")
    else:
        st.error("User not found")