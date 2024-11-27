import streamlit as st
from navigation import make_sidebar

make_sidebar()

# Check if the user is logged in and retrieve the username from session state
if "logged_in" in st.session_state and st.session_state.logged_in:
    username = st.session_state.username  # Retrieve the username from session state
else:
    st.error("You need to log in first.")
    st.stop()  # Stop the execution of the page if not logged in

# Welcome message with the username
st.markdown(f"<h1>Welcome to Sakila Analytics app, <span style='color:orange'>{username}</span>!</h1>", unsafe_allow_html=True)

st.markdown(
    """
    Dive into comprehensive insights and visualizations of the Sakila Database, a sample database representing a fictional DVD rental store. This app is designed to help you explore the data, uncover trends, and make data-driven decisions.

    **Key Features:**\n
    - **Customer Insights**: Analyze customer demographics and rental behaviors.\n
    - **Rental Trends**: Discover patterns in movie rentals across different time periods.\n
    - **Store Performance**: Compare the performance of different store locations.\n
    - **Inventory Analysis**: Evaluate the availability and popularity of movies.\n
    - **Revenue Breakdown**: Track revenue trends and identify top contributors.\n

    Whether you're a data enthusiast, business analyst, or just curious about how analytics works, this app offers an engaging way to explore the Sakila dataset.

    Let’s get started and uncover the story hidden in the data! 🕵️‍♂️📈
    """
)



st.write("# About the Sakila Database! 🎦")

st.markdown(
    '''
    The **Sakila Database** is a sample database provided by MySQL that is designed to help users learn and practice database management, querying, and relational database design. It is based on a fictional DVD rental store and is structured to showcase common relational database features, including normalization, relationships, and use of foreign keys.

    ### Key Features of the Sakila Database:

    #### **Entities and Relationships**:
    - The database includes multiple entities representing aspects of a DVD rental store, such as **customers**, **films**, **rentals**, **payments**, **staff**, and **stores**.
    - Relationships between these entities are modeled using foreign keys and join tables, demonstrating many-to-one and many-to-many relationships.

    #### **Schema Overview**:
    - **Customer**: Contains customer data such as names, addresses, and active status.
    - **Film**: Stores information about the films available for rental, including titles, descriptions, release years, and rental rates.
    - **Actor**: Represents actors featured in films, linked to the films they appear in.
    - **Inventory**: Tracks copies of films available at each store location.
    - **Rental**: Records rental transactions, including rental and return dates.
    - **Payment**: Logs payments made by customers for their rentals.
    - **Staff**: Holds information about the store staff members.

    #### **Designed for Query Practice**:
    - Provides realistic scenarios for users to practice SQL queries like **SELECT**, **JOIN**, **GROUP BY**, **HAVING**, and more.
    - Contains comprehensive data to experiment with advanced SQL concepts, including subqueries, views, and stored procedures.

    #### **Sample Data**:
    - Includes a rich dataset that mimics real-world data, making it suitable for training and development purposes.

    #### **Normalization**:
    - The database is normalized to ensure efficient data storage and retrieval, demonstrating proper relational database design principles.

    The **Sakila Database** is widely used for educational purposes, tutorials, and testing SQL skills in a controlled environment.
    ''')