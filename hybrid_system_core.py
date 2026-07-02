import ollama
import pandas as pd
import joblib
import json
import re
import plotly.express as px
from sqlalchemy import create_engine
from config import DATABASE_URL

# Database connection
engine = create_engine(DATABASE_URL)

# Intent Classification
def classify_intent(question):
    prompt = f"""
You are an intent classifier for a medical analytics system.

Classify the query into ONE of the two categories:

SQL:
- Descriptive queries
- Aggregations (count, average, sum)
- Listing or filtering records
- Historical analysis
- Filtering
- Database queries

PREDICTION:
- Forecasting
- Risk estimation
- Future outcome prediction
- Probability estimation

Respond with ONLY one word: SQL or PREDICTION.

Query:
{question}
"""
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"].strip()

# Visualization detection
def needs_visualization(question):
    keywords = ["plot", "chart", "graph", "distribution", "trend", "visualize"]
    return any(word in question.lower() for word in keywords)

# Visualization
def generate_visualization(df):

    if df is None or df.empty:
        return "No data available."

    # CASE 1 — category + numeric → bar chart
    if df.shape[1] == 2:

        col1, col2 = df.columns

        if pd.api.types.is_numeric_dtype(df[col1]):
            y_col = col1
            x_col = col2
        else:
            x_col = col1
            y_col = col2

        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title="Query Result Visualization",
            text_auto=True
        )

        fig.update_layout(
            height=400,
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=y_col.replace("_", " ").title(),
            margin=dict(l=20, r=20, t=60, b=20)
        )

        return fig

    # CASE 2 — single numeric column → histogram
    if df.shape[1] == 1 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):

        col = df.columns[0]

        fig = px.histogram(
            df,
            x=col,
            nbins=20,
            title=f"Distribution of {col.replace('_', ' ').title()}"
        )

        fig.update_layout(
            height=400,
            xaxis_title=col.replace("_", " ").title(),
            yaxis_title="Frequency",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        return fig

    # CASE 3 — fallback → return table
    return df

# Full Data Schema
SCHEMA_FULL = """
Table: diabetic_flat
Columns:
encounter_id (int),
patient_nbr (int),
race (text),
gender (text),
age (text),
weight (text),
admission_type_id (int),
discharge_disposition_id (int),
admission_source_id (int),
time_in_hospital (int),
payer_code (text),
medical_specialty (text),
num_lab_procedures (int),
num_procedures (int),
num_medications (int),
number_outpatient (int),
number_emergency (int),
number_inpatient (int),
diag_1 (text),
diag_2 (text),
diag_3 (text),
number_diagnoses (int),
max_glu_serum (text),
"A1Cresult" (text),
metformin (text),
repaglinide (text),
nateglinide (text),
chlorpropamide (text),
glimepiride (text),
acetohexamide (text),
glipizide (text),
glyburide (text),
tolbutamide (text),
pioglitazone (text),
rosiglitazone (text),
acarbose (text),
miglitol (text),
troglitazone (text),
tolazamide (text),
examide (text),
citoglipton (text),
insulin (text),
glyburide_metformin (text),
glipizide_metformin (text),
glimepiride_pioglitazone (text),
metformin_rosiglitazone (text),
metformin_pioglitazone (text),
"change" (text),
"diabetesMed" (text),
readmitted (text)
"""
# Get relevant columns from question
def extract_relevant_columns(question):
    question_lower = question.lower()

    all_cols_query = """
    SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'diabetic_flat'
    """

    df_cols = pd.read_sql(all_cols_query, engine)
    all_columns = df_cols.iloc[:, 0].tolist()

    relevant = []

    for col in all_columns:
        if col.lower() in question_lower:
            relevant.append(col)

    return relevant

# Get distinct values for categorical columns
def get_distinct_values(column_name, limit=10):
    try:
        query = f"""
        SELECT "{column_name}", COUNT(*) AS freq
        FROM diabetic_flat
        GROUP BY "{column_name}"
        ORDER BY freq DESC
        LIMIT {limit};
        """
        df = pd.read_sql(query, engine)
        values = df.iloc[:, 0].dropna().astype(str).tolist()
        return values
    
    except Exception as e:
        print(e)
        return []

# Provide hint for llama (valid value in dataset) to improve SQL generation
def build_value_hints(question):
    relevant_cols = extract_relevant_columns(question)
    hints = ""

    for col in relevant_cols:
        values = get_distinct_values(col, limit=5)
        if values:
            hints += f"\nValid values for {col}: {values}"

    return hints

# Text-to-SQL Query Generation
def llama_text_to_sql(question):

    value_hints = build_value_hints(question)

    prompt = f"""
You are an expert system that converts natural language questions into SQL.
Schema:
{SCHEMA_FULL}
Distinct values hints based on question context:
{value_hints}

Important:
Use column names EXACTLY as written in the SCHEMA_FULL that provided.

If a column name is enclosed in double quotes, preserve the double quotes exactly in the generated SQL.
Do not remove or change the capitalization of quoted identifiers.

Do not invent or modify column names.
If the question is unrelated to the dataset or cannot be answered using the schema,
return: INVALID_QUERY

Important Semantic Rules:
- In this dataset, readmitted values are:
  '<30' = readmitted within 30 days
  '>30' = readmitted after 30 days
  'NO' = not readmitted
- When a question says "were readmitted", it means readmitted <> 'NO'.
- If question asks for distribution, use GROUP BY and COUNT(*)

Important SQL Behavior Rules:

1. If user asks to list records or show patients,
   use SELECT * with LIMIT.

2. If user asks for distribution of a numeric column,
   select that column directly (no COUNT or GROUP BY).

3. If user asks for counts by category, use GROUP BY with COUNT(*).

General Rules:
- If using GROUP BY, include grouped column in SELECT.
- Use COUNT(*) instead of COUNT(column).
- Output ONLY valid PostgreSQL SQL.
- End the query with a semicolon.

Question:
{question}

SQL:
"""
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"].strip()

# fix SQL if GROUP BY without column in SELECT
def fix_groupby_sql(sql):

    if "GROUP BY" in sql.upper():

        group_col = group_col = re.search(r'GROUP BY\s+("?[\w]+"?)', sql, re.IGNORECASE)

        if group_col:
            col = group_col.group(1)

            select_part = sql.split("FROM")[0]

            if col not in select_part:

                sql = sql.replace(
                    "SELECT",
                    f"SELECT {col},",
                    1
                )

    return sql

# SQL Query Execution
def execute_sql(sql):
    try:
        return pd.read_sql(sql, engine)
    except:
        return None
    
# Hybrid Combination
def handle_query(question):

    intent = classify_intent(question)

    if intent == "SQL":

        sql = llama_text_to_sql(question)
        sql = fix_groupby_sql(sql)
        print("Generated SQL:", sql)

        df = execute_sql(sql)

        if df is None:
            return "Invalid SQL generated. Please try rephrasing your question or state column names clearly."

        elif needs_visualization(question):
            return generate_visualization(df)

        return df

    elif intent == "PREDICTION":

        return "Prediction and analytic report have been generated, please refer to the corresponding pages."
    
    return "⚠ The system could not process this query. Please try rephrasing."