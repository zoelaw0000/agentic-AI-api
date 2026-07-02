import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from config import DATABASE_URL

MODEL_PATH = "readmitted_model_compressed.pkl"

PREDICTION_THRESHOLD = 0.13

model = None

def get_model():
    global model

    if model is None:
        print("Loading model...")
        model = joblib.load(MODEL_PATH)
        print("Model loaded.")

    return model


def prepare_prediction_data(df):

    columns_to_drop = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "weight",
        "max_glu_serum",
        "A1Cresult"
    ]

    prediction_df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    ).copy()

    #convert pandas missing values in database to numpy missing values
    prediction_df = prediction_df.replace({pd.NA: np.nan})

    return prediction_df

def predict_batch(df):

    prediction_df = prepare_prediction_data(df)

    model = get_model()

    probabilities = model.predict_proba(
        prediction_df
    )[:, 1]

    risk_levels = [
        "High Risk"
        if p >= PREDICTION_THRESHOLD
        else "Low Risk"
        for p in probabilities
    ]

    predictions = [
        int(p >= PREDICTION_THRESHOLD)
        for p in probabilities
    ]

    result = df.copy()

    result["readmission_probability"] = probabilities

    result["prediction"] = predictions

    result["risk_level"] = risk_levels

    return result