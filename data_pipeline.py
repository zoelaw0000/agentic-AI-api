import pandas as pd
from sqlalchemy import create_engine, text
from ml_model import predict_batch
from config import DATABASE_URL

# System database
engine = create_engine(DATABASE_URL)

# Schema validation
EXPECTED_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide_metformin",
    "glipizide_metformin",
    "glimepiride_pioglitazone",
    "metformin_rosiglitazone",
    "metformin_pioglitazone",
    "change",
    "diabetesMed",
    "readmitted"
]

def validate_schema(df):

    # Standardize column names before validation
    df.columns = [c.strip().replace("-", "_") for c in df.columns]

    uploaded_columns = set(df.columns)
    expected_columns = set(EXPECTED_COLUMNS)

    if uploaded_columns != expected_columns:
        missing = expected_columns - uploaded_columns
        extra = uploaded_columns - expected_columns

        return False, {
            "missing_columns": list(missing),
            "extra_columns": list(extra)
        }

    return True, "Schema valid."

NUMERIC_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

# Data Cleaning
def clean_data(df):

    # Replace missing values with NaN (safe for database insertion)
    df = df.replace(["?", "None", "", "NA", "null"], pd.NA)

    # Remove completely empty rows
    df = df.dropna(how="all")

    #drop rows with missing encounter_id as it's the primary key      
    df = df.dropna(subset=["encounter_id"])

    # Strip whitespace for object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Enforce numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # sanity filtering
    if "time_in_hospital" in df.columns:
        df = df[df["time_in_hospital"] >= 0]

    if "num_medications" in df.columns:
        df = df[df["num_medications"] >= 0]
        
    return df

# Duplicate Detection
def remove_existing_duplicates(df):

    existing_ids = pd.read_sql(
        "SELECT encounter_id FROM diabetic_flat;",
        engine
    )

    existing_set = set(existing_ids["encounter_id"].tolist())

    before = len(df)

    df = df[~df["encounter_id"].isin(existing_set)]

    after = len(df)

    removed = before - after

    return df, removed

# Insert new data into database
def insert_data(df):

    df.to_sql(
        "diabetic_flat",
        engine,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi"
    )

def save_prediction_result(df):

    with engine.begin() as conn:
        conn.execute(
            text("TRUNCATE TABLE patient_predictions")
        )

    df.to_sql(
        "patient_predictions",
        engine,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi"
    )

# user upload function
def upload_and_process(file):

    df = pd.read_csv(file)

    # Schema validation
    valid, message = validate_schema(df)
    if not valid:
        return {"status": "error", "details": message}

    # Clean
    df = clean_data(df)

    # Remove duplicates
    df, removed = remove_existing_duplicates(df)

    if len(df) == 0:
        return {
            "status": "warning",
            "message": "All rows already exist in database."
        }

    # Insert
    insert_data(df)

    prediction_result = predict_batch(df)

    # Save latest prediction results
    save_prediction_result(prediction_result)

    high_risk_count = (
        prediction_result["risk_level"] == "High Risk"
    ).sum()

    high_risk_rate = round(
        high_risk_count / len(prediction_result) * 100,
        2
    )

    avg_probability = round(
        prediction_result["readmission_probability"].mean(),
        4
    )

    return {
        "status": "success",
        "inserted_rows": len(df),
        "duplicates_removed": removed,
        "prediction_model_status": "Pre-trained model loaded.",

        "high_risk_count": int(high_risk_count),
        "high_risk_rate": high_risk_rate,
        "average_probability": avg_probability,

        "prediction_result": prediction_result

    }