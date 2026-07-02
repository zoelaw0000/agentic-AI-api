import pandas as pd
from fastapi import FastAPI, UploadFile, File
from sqlalchemy import create_engine
from pydantic import BaseModel
from data_pipeline import upload_and_process
from analytics_engine import generate_analytics_report
import config
from hybrid_system_core import (build_value_hints, fix_groupby_sql, execute_sql, generate_visualization)

engine = create_engine(config.DATABASE_URL)

#uvicorn main:app --reload
#http://127.0.0.1:8000/
#http://127.0.0.1:8000/docs
app = FastAPI(
    title="Hybrid Medical Data Management API",
    version="1.0"
)

@app.get("/")
def home():
    return {
        "message": "Hybrid Medical Data Management API is running."
    }

@app.post("/process-data")
async def process_data(file: UploadFile = File(...)):

    result = upload_and_process(file.file)

    if result["status"] == "success":

        return {
            "status": "success",
            "inserted_rows": int(result["inserted_rows"]),
            "duplicates_removed": int(result["duplicates_removed"]),
            "prediction_model_status": result["Prediction_model_status"],
            "high_risk_count": int(result["high_risk_count"]),
            "high_risk_rate": float(result["high_risk_rate"]),
            "average_probability": float(result["average_probability"])
        }

    return result

@app.post("/analytics")
def analytics():

    df = pd.read_sql(
        "SELECT * FROM patient_predictions ORDER BY encounter_id",
        engine
    )

    if df.empty:
        return {
            "status": "error",
            "message": "No prediction data found."
        }

    report = generate_analytics_report(df)

    return report

class QuestionRequest(BaseModel):
    question: str

@app.post("/value-hints")
def value_hints(request: QuestionRequest):

    hints = build_value_hints(request.question)

    return {
        "status": "success",
        "value_hints": hints
    }

class SQLRequest(BaseModel):
    sql: str

@app.post("/execute-query")
def execute_query_api(request: SQLRequest):

    sql = fix_groupby_sql(request.sql)

    df = execute_sql(sql)

    if df is None:
        return {
            "status": "error",
            "message": "Invalid SQL."
        }

    return {
        "status": "success",
        "sql": sql,
        "row_count": int(len(df)),
        "data": df.to_dict(orient="records")
    }

class VisualizationRequest(BaseModel):
    data: list

@app.post("/generate-visualization")
def generate_visualization_api(request: VisualizationRequest):

    df = pd.DataFrame(request.data)

    fig = generate_visualization(df)

    if isinstance(fig, pd.DataFrame):
        return {
            "status": "table",
            "data": fig.to_dict(orient="records")
        }

    return {
        "status": "chart",
        "figure": fig.to_json()
    }