import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def get_summary_metrics(df):

    metrics = {
        "uploaded_cases": int(len(df)),

        "high_risk_patients": int(
            (df["risk_level"] == "High Risk").sum()
        ),

        "average_risk_score": float(
            df["readmission_probability"].mean()
        ),

        "average_stay": float(
            df["time_in_hospital"].mean()
        ),

        "average_medications": float(
            df["num_medications"].mean()
        )
    }

    return metrics

def get_age_distribution_chart(df):

    age_counts = (
        df["age"]
        .value_counts()
        .sort_index()
    )

    fig = px.bar(
        x=age_counts.index,
        y=age_counts.values,
        labels={
            "x": "Age Group",
            "y": "Number of Patients"
        },
        title="Patient Age Distribution"
    )

    return fig

def get_gender_distribution_chart(df):

    gender_counts = df["gender"].value_counts()

    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title="Patient Gender Distribution",
        hole=0.4
    )

    return fig

def get_race_distribution_chart(df):

    race_counts = df["race"].fillna("Unknown").value_counts()

    fig = px.bar(
        x=race_counts.values,
        y=race_counts.index,
        orientation="h",
        labels={
            "x": "Number of Patients",
            "y": "Race"
        },
        title="Patient Race Distribution"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    return fig

def get_hospital_stay_chart(df):

    fig = px.histogram(
        df,
        x="time_in_hospital",
        nbins=14,
        title="Hospital Stay Duration",
        labels={
            "time_in_hospital": "Length of Stay (Days)"
        }
    )

    fig.update_layout(
        yaxis_title="Number of Patients"
    )

    return fig

def get_medication_chart(df):

    fig = px.histogram(
        df,
        x="num_medications",
        nbins=20,
        title="Number of Medications",
        labels={
            "num_medications": "Medication Count"
        }
    )

    fig.update_layout(
        yaxis_title="Number of Patients"
    )

    return fig

def get_risk_distribution_chart(df):

    risk_counts = df["risk_level"].value_counts()

    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Predicted Readmission Risk",
        hole=0.4,
        color=risk_counts.index,
        color_discrete_map={
            "High Risk": "#EF4444",
            "Low Risk": "#10B981"
        }
    )

    return fig

def generate_patient_clusters(df):

    features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"
    ]

    cluster_df = df[features].copy()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(cluster_df)

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(scaled_data)

    result_df = df.copy()

    result_df["cluster"] = clusters

    return result_df

def get_cluster_profiles(df):

    cluster_df = generate_patient_clusters(df)

    profile = (
        cluster_df
        .groupby("cluster")[
            [
                "time_in_hospital",
                "num_lab_procedures",
                "num_medications",
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
                "number_diagnoses"
            ]
        ]
        .mean()
        .round(2)
    )

    return profile

def get_cluster_heatmap(df):

    profile = get_cluster_profiles(df)

    fig = px.imshow(
        profile,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Patient Cluster Profiles"
    )

    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Cluster"
    )

    return fig

def get_cluster_risk_chart(df):

    cluster_df = generate_patient_clusters(df)

    rates = (
        cluster_df
        .groupby("cluster")["risk_level"]
        .apply(
            lambda x: (x == "High Risk").mean() * 100
        )
        .reset_index(name="high_risk_rate")
    )

    fig = px.bar(
        rates,
        x="cluster",
        y="high_risk_rate",
        text="high_risk_rate",
        title="High-Risk Rate by Patient Cluster"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="High-Risk Rate (%)"
    )

    return fig

# for AI-generated insight
def generate_insights_input(df):

    cluster_df = generate_patient_clusters(df)

    cluster_risk = (
        cluster_df
        .groupby("cluster")["risk_level"]
        .apply(
            lambda x: (x == "High Risk").mean() * 100
        )
        .to_dict()
    )

    highest_risk_cluster = max(
        cluster_risk,
        key=cluster_risk.get
    )

    insights = {
        "total_cases": len(df),

        "high_risk_rate": (
            (df["risk_level"] == "High Risk")
            .mean() * 100
        ),

        "average_stay": (
            df["time_in_hospital"].mean()
        ),

        "average_medications": (
            df["num_medications"].mean()
        ),

        "cluster_risk": cluster_risk,

        "highest_risk_cluster": highest_risk_cluster,

        "highest_risk_cluster_rate": (
            cluster_risk[highest_risk_cluster]
        )
    }

    return insights

def generate_analytics_report(df):

    summary_metrics = get_summary_metrics(df)

    age_chart = get_age_distribution_chart(df)
    gender_chart = get_gender_distribution_chart(df)
    race_chart = get_race_distribution_chart(df)

    stay_chart = get_hospital_stay_chart(df)
    medication_chart = get_medication_chart(df)

    risk_chart = get_risk_distribution_chart(df)

    cluster_heatmap = get_cluster_heatmap(df)
    cluster_risk = get_cluster_risk_chart(df)

    insight_input = generate_insights_input(df)

    return {
        "status": "success",

        "metrics": summary_metrics,

        "charts": {
            "age_distribution": pio.to_json(age_chart),
            "gender_distribution": pio.to_json(gender_chart),
            "race_distribution": pio.to_json(race_chart),
            "hospital_stay": pio.to_json(stay_chart),
            "medication_distribution": pio.to_json(medication_chart),
            "risk_distribution": pio.to_json(risk_chart),
            "cluster_heatmap": pio.to_json(cluster_heatmap),
            "cluster_risk": pio.to_json(cluster_risk)
        },

        "insight_input": insight_input
    }