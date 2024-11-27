import streamlit as st
from navigation import make_sidebar
from pages.analytics import run_query

make_sidebar()

st.title("Sakila Database Viewer 📜")

table = st.selectbox("**Select a table:**", ["actor", "film", "customer", "rental", "payment"])

# Multiselect for columns
columns_df = run_query(f"PRAGMA table_info({table})")

columns = columns_df["name"].tolist()

# Multiselect for selecting columns
selected_columns = st.multiselect(
    "**Select columns:**",
    columns,
    default=[]
)

# Display selected columns (optional)
st.write(f"Selected columns: {selected_columns}")

# Optional WHERE clause input
where_clause = st.text_input("**Enter a WHERE clause (optional):**", value="")

# Optional GROUP BY input
# group_by = st.selectbox("**Select columns to GROUP BY (optional):**", ["None"] + columns)
group_by = st.multiselect(
    "**Select columns to GROUP BY (optional):**",
    columns,  # Show the columns available for GROUP BY
    default=[]
)

# Constructing and running the query when the button is pressed
if st.button("Run Query"):
    query = f"SELECT {', '.join(selected_columns) if selected_columns else '*'} FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    if group_by != "None":
        query += f" GROUP BY {', '.join(group_by)}"

    # Display the constructed query for debugging
    st.write(f"Executing query: {query}")

    # Run the query and display the result
    result_df = run_query(query)
    
    if not result_df.empty:
        st.success("Query executed successfully!")
        st.dataframe(result_df)

        # Option to export the result to CSV
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{table}_query_results.csv",
            mime="text/csv"
        )
    else:
        st.error("No results returned.")