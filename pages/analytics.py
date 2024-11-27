import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from navigation import make_sidebar

make_sidebar()

def run_query(query):

    conn = sqlite3.connect("./sakila_db/sakila_master.db")
    df = pd.read_sql(query, conn)
    conn.close()
    return df

st.title('Sales Overview Analysis 📉')

# Total revenue
st.subheader('Total Revenue')
total_revenue = run_query ("SELECT SUM(amount) AS total_revenue FROM payment;")
st.write(f"Total Revenue: {total_revenue['total_revenue'].iloc[0]:,.2f}")

# Monthly revenue Trends
st.subheader("Monthly Revenue Trends")
monthly_revenue = run_query("""
    SELECT strftime('%Y-%m', payment_date) AS month, SUM(amount) AS revenue
    FROM payment
    GROUP BY month
    ORDER BY month;
""")
st.line_chart(monthly_revenue.set_index('month'))

# Top 10 highest grossing films
st.subheader("Top 10 Highest Grossing Films")
top_films = run_query("""
    SELECT f.title, SUM(p.amount) AS revenue
    FROM film f
    JOIN inventory i ON f.film_id = i.film_id
    JOIN rental r ON i.inventory_id = r.inventory_id
    JOIN payment p ON r.rental_id = p.rental_id
    GROUP BY f.title
    ORDER BY revenue DESC
    LIMIT 10;
""")
fig = px.bar(
    top_films,
    x='title',
    y='revenue',
    color='title',  # Use the title column to assign unique colors
    title="Top 10 Highest Grossing Films",
    labels={'title': 'Film Title', 'revenue': 'Revenue'},
)
fig.update_layout(
    xaxis_title="Film Title",
    yaxis_title="Revenue",
    showlegend=False,  # Hide the legend since each bar is uniquely colored
    template="plotly_dark"  # Optional: use a dark template
)
st.plotly_chart(fig)

# Top Revenue Generating Customers
st.subheader("Top Revenue Generating Customers")
top_customers = run_query("""
    SELECT c.first_name || ' ' || c.last_name AS customer_name, SUM(p.amount) AS total_spent
    FROM customer c
    JOIN rental r ON c.customer_id = r.customer_id
    JOIN payment p ON r.rental_id = p.rental_id
    GROUP BY customer_name
    ORDER BY total_spent DESC
    LIMIT 10;
""")
st.table(top_customers)

# Average Rental Duration and Revenue per Category
st.subheader("Average Rental Duration and Revenue per Category")
category_revenue = run_query("""
    SELECT c.name AS category, AVG(julianday(r.return_date) - julianday(r.rental_date)) AS avg_rental_duration,
           SUM(p.amount) AS total_revenue
    FROM category c
    JOIN film_category fc ON c.category_id = fc.category_id
    JOIN film f ON fc.film_id = f.film_id
    JOIN inventory i ON f.film_id = i.film_id
    JOIN rental r ON i.inventory_id = r.inventory_id
    JOIN payment p ON r.rental_id = p.rental_id
    GROUP BY c.name
    ORDER BY total_revenue DESC;
""")
st.table(category_revenue)

