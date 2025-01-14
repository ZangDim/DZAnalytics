import streamlit as st
from pygments import highlight
from pygments.lexers.sql import SqlLexer
from pygments.formatters.html import HtmlFormatter
from navigation import make_sidebar
from pages.analytics import run_query

make_sidebar()

st.title("Sakila Database Viewer 📜")

# Display schema
if st.button("Show schema"):
    st.image('img/sakila_schema.png', caption="Sakila Database Schema", use_column_width=True)

st.markdown("---")

# Allow the user to write a SQL query
st.markdown("### Write Your SQL Query:")
custom_query = st.text_area(
    "Enter your SQL query below:", 
    placeholder="Write your SQL query here (e.g., SELECT * FROM customer LIMIT 10);",
    height=300  # Larger field for complex queries
)

# Live SQL Preview with Syntax Highlighting
st.markdown("#### Live SQL Query Preview:")
if custom_query.strip():
    formatter = HtmlFormatter(style="monokai", full=False, noclasses=True)
    highlighted_query = highlight(custom_query, SqlLexer(), formatter)
    st.markdown(f"<div style='font-family: monospace;'>{highlighted_query}</div>", unsafe_allow_html=True)
else:
    st.warning("Start typing your SQL query above to see the live preview by pressing Cntrl + Enter to apply .")

# Execute the query
if st.button("Run Query"):
    if custom_query.strip():  # Validate non-empty query
        try:
            result_df = run_query(custom_query)
            if result_df is not None and not result_df.empty:
                st.success("Query executed successfully!")
                st.dataframe(result_df)

                # Option to download results as CSV
                csv = result_df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="custom_query_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results returned. Please check your query.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid SQL query.")

st.markdown("---")

# Tips section
st.info("""
💡 **Query Tips**:
- Use the schema above to identify table relationships and keys.
- Make sure your SQL syntax is valid to avoid errors.
- Example query:
```sql
SELECT c.customer_id, c.first_name, c.last_name,
       COUNT(r.rental_id) AS rental_count,
       SUM(p.amount) AS total_spent,
       AVG(f.rental_duration) AS avg_rental_duration,
       COUNT(DISTINCT f.film_id) AS distinct_films_rented,
       c.store_id, c.email
FROM customer c
LEFT JOIN rental r ON c.customer_id = r.customer_id
LEFT JOIN payment p ON c.customer_id = p.customer_id
LEFT JOIN inventory i ON i.inventory_id = r.inventory_id
LEFT JOIN film f ON f.film_id = i.film_id
GROUP BY c.customer_id;
        """)







# import streamlit as st
# from navigation import make_sidebar
# from pages.analytics import run_query

# make_sidebar()

# st.title("Sakila Database Viewer 📜")

# # Create a button that shows the schema image when pressed
# if st.button("Show schema"):
#     st.image('img/sakila_schema.png', caption="Sakila Database Schema", use_column_width=True)

# def display_tables():
#     query = "SELECT name FROM sqlite_master WHERE type='table';"
#     tables_df = run_query(query)
#     tables_df = tables_df['name'].tolist()
#     return tables_df

# # Step 1: Select the primary table
# table = st.selectbox("**Select the primary table:**", display_tables())

# # Step 2: Display columns for the selected table
# if table:
#     # Fetch and display columns from the primary table
#     columns_df = run_query(f"PRAGMA table_info({table})")
#     columns = columns_df["name"].tolist()

#     selected_columns = st.multiselect(
#         "**Select columns from primary table:**",
#         columns,
#         default=[],  # Default to some basic columns
#         format_func=lambda col: f"{table}.{col}"  # Display fully qualified names
#     )

#     # Ensure selected columns are fully qualified
#     selected_columns = [f"{table}.{col}" for col in selected_columns]

#     # Initialize storage for aggregation types (COUNT, AVG, SUM, etc.)
#     aggregation_types = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']

#     # Step 3: Allow user to select aggregation function for each column
#     aggregation_functions = {}
#     for col in selected_columns:
#         aggregation_type = st.selectbox(
#             f"**Select aggregation function for {col}:**",
#             ['None'] + aggregation_types,  # None option for non-aggregated columns
#             key=f"agg_{col}"
#         )
#         aggregation_functions[col] = aggregation_type

#     # Initialize storage for joined tables
#     joined_tables = []

#     # Types of join
#     join_types = ['JOIN', 'LEFT JOIN', 'RIGHT JOIN']

#     # Allow adding multiple JOIN clauses dynamically
#     add_more = True
#     while add_more:
#         # Select a table to join
#         join_table = st.selectbox(
#             "**Select a table to JOIN (optional):**",
#             display_tables() + ["None"],
#             key=f"join_table_{len(joined_tables)}"
#         )

#         if join_table != "None":
#             # Fetch columns for the join table
#             join_columns_df = run_query(f"PRAGMA table_info({join_table})")
#             join_columns = join_columns_df["name"].tolist()

#             # Prompt for the JOIN condition
#             join_condition = st.text_input(
#                 f"**Enter the JOIN condition for {join_table}:**",
#                 value="",
#                 key=f"join_condition_{len(joined_tables)}"
#             )

#             join_type = st.selectbox(
#                 f"**Select type of JOIN for {join_table}:**",
#                 join_types,  # This will provide the options for JOIN, LEFT JOIN, RIGHT JOIN
#                 index=1  # Default to 'LEFT JOIN'
#             )

#             # Multiselect for selecting columns from the join table
#             selected_join_columns = st.multiselect(
#                 f"**Select columns from {join_table}:**",
#                 join_columns,
#                 default=[],
#                 format_func=lambda col: f"{join_table}.{col}",
#                 key=f"select_columns_{len(joined_tables)}"
#             )

#             # Store the join table, condition, and selected columns along with the join type
#             if join_condition and selected_join_columns:
#                 joined_tables.append((join_table, join_condition, selected_join_columns, join_type))

#         # Add an option to stop adding JOIN clauses
#         add_more = st.checkbox("**Add another JOIN?**", key=f"add_more_{len(joined_tables)}")

#         # Example Output: Print the selected columns and joins
#         st.write("Primary Table Columns:", selected_columns)
#         st.write("Joined Tables:", joined_tables)

#     # Step 4: Specify GROUP BY columns
#     group_by = st.multiselect(
#         "**Select columns to GROUP BY (optional):**",
#         [f"{col}" for col in selected_columns],
#         default=[]  
#     )

#     # Step 5: Construct and run the query when the button is pressed
#     if st.button("Run Query"):
#         try:
#             # Construct the SELECT clause with aggregation functions if selected
#             select_clause = []
#             for col in selected_columns:
#                 agg_function = aggregation_functions[col]
#                 if agg_function == 'None':
#                     select_clause.append(f"{col}")  # Non-aggregated column
#                 else:
#                     select_clause.append(f"{agg_function}({col}) AS {agg_function.lower()}_{col.split('.')[-1]}")

#             # Build the base query
#             query = f"SELECT {', '.join(select_clause)} FROM {table}"

#             # Add the JOIN clauses
#             for join_table, join_condition, _, join_type in joined_tables:
#                 query += f" {join_type} {join_table} ON {join_condition}"

#             # Add the GROUP BY clause
#             if group_by:
#                 query += f" GROUP BY {', '.join(group_by)}"

#             # Display the constructed query for debugging
#             st.code(query, language='sql')

#             # Run the query and display the result
#             result_df = run_query(query)
#             if result_df is not None and not result_df.empty:
#                 st.success("Query executed successfully!")
#                 st.dataframe(result_df)

#                 # Option to export the result to CSV
#                 csv = result_df.to_csv(index=False, encoding='utf-8')
#                 st.download_button(
#                     label="Download as CSV",
#                     data=csv,
#                     file_name=f"{table}_query_results.csv",
#                     mime="text/csv"
#                 )
#             else:
#                 st.warning("No results returned.")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
