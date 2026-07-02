import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sqlalchemy import create_engine
from streamlit_option_menu import option_menu
import os

from hybrid_system_core import handle_query
from data_pipeline import upload_and_process
from analytics_engine import (get_summary_metrics, get_age_distribution_chart, 
                              get_gender_distribution_chart, get_race_distribution_chart,
                              get_hospital_stay_chart, get_medication_chart,
                              get_risk_distribution_chart, get_cluster_heatmap, 
                              get_cluster_risk_chart, generate_insights_input
                              )
from config import DATABASE_URL

# Page configuration
st.set_page_config(
    page_title="Hybrid Medical Data Management System",
    page_icon="🏥",
    layout="wide"
)

# Database Connection
engine = create_engine(DATABASE_URL)

st.markdown("""
<style>

/* Metric Cards */
[data-testid="stMetric"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("🏥 Hybrid Medical Data Management & Analytics System")
st.caption(
    "Conversational Healthcare Data Analysis and Readmission Prediction"
)

st.divider()

# Sidebar navigation

with st.sidebar:

    page = option_menu(
        "Navigation",
        [
            "Dashboard",
            "Upload Data",
            "Prediction",
            "Analytic Report",
            "Natural Language Analysis"
        ],
        icons=[
            "house",
            "cloud-upload",
            "activity",
            "bar-chart",
            "robot"
        ],
        default_index=0
    )

# Page routing
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
if page == "Dashboard":

    st.header("System Overview")

    st.subheader("System Status")

    col1, col2, col3 = st.columns(3)

    # Database record count
    with col1:
        try:
            df = pd.read_sql(
                "SELECT COUNT(*) as total FROM diabetic_flat;",
                engine
            )
            st.metric("Database Records", int(df["total"][0]))
        except:
            st.error("Database connection failed")

    # Model availability
    with col2:
        st.metric(
            "Prediction Model",
            "Loaded" if os.path.exists("readmitted_model.pkl") else "Missing"
        )

    with col3:
        st.metric(
            "Llama 3.1:8B Engine",
            "Available"
        )

    # -----------------------------
    # Dataset preview
    # -----------------------------

    try:
        preview = pd.read_sql(
            "SELECT * FROM diabetic_flat;",
            engine
        )

        st.subheader("Dataset Statistics")

        col1,col2,col3,col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Records",
                len(preview)
            )

        with col2:
            st.metric(
                "Avg Hospital Stay",
                f"{preview['time_in_hospital'].mean():.1f}"
            )

        with col3:
            st.metric(
                "Avg Medications",
                f"{preview['num_medications'].mean():.1f}"
            )

        with col4:
            st.metric(
                "Avg Diagnoses",
                f"{preview['number_diagnoses'].mean():.1f}"
            )
        
        tab1, tab2 = st.tabs([
            "📊 Analytics Dashboard",
            "📋 Dataset Preview"
        ])

        with tab1:

            col1, col2 = st.columns(2)

            with col1:

                fig = px.histogram(
                    preview,
                    x="gender",
                    title="Gender Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with col2:

                fig = px.histogram(
                    preview,
                    x="time_in_hospital",
                    title="Hospital Stay Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with tab2:

            st.dataframe(
                preview.head(100)
            )

    except:
        st.info("No data available yet.")
    

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

elif page == "Upload Data":

    st.header("Upload Medical Dataset")

    st.markdown("""
    ### Processing Pipeline

    📂 Upload Dataset  
    ➡ Validate Dataset  
    ➡ Automated Preprocessing  
    ➡ Database Storage  
    ➡ Readmission Prediction  
    ➡ AI Analytics Generation  
    """)

    st.write(
        "Upload a CSV file. The system will automatically clean, validate, "
        "insert the data into the database, predict readmission risks, and show analytic report."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        st.info("File uploaded successfully.")

        if st.button("Process and Insert Data"):

            with st.spinner("Processing dataset..."):

                result = upload_and_process(uploaded_file)

            # Display pipeline result
            if result["status"] == "success":

                st.session_state["upload_result"] = result

                st.session_state["analytics_df"] = result["prediction_result"]

                st.success("✅ Dataset processed successfully")

                col1,col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Rows Inserted",
                        result["inserted_rows"]
                    )

                with col2:
                    st.metric(
                        "Duplicates Removed",
                        result["duplicates_removed"]
                    )

                with st.container(border=True):

                    st.success(
                        "✅ Dataset processed successfully. "
                        "Please navigate to the Prediction page & Analytic Report page to view prediction and analytics."
                    )

            elif result["status"] == "warning":

                st.warning(result["message"])

            else:

                st.error("Upload failed.")
                st.json(result["details"])

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == "Prediction":

    st.subheader("📊 Prediction Summary")

    if "upload_result" not in st.session_state:

        st.info(
            "No prediction results available. "
            "Please upload a dataset first."
        )

    else:

        result = st.session_state["upload_result"]

        with st.container(border=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "High-Risk Patients",
                    result["high_risk_count"]
                )

            with col2:
                st.metric(
                    "High-Risk Rate",
                    f"{result['high_risk_rate']}%"
                )

            with col3:
                st.metric(
                    "Average Risk Score",
                    f"{result['average_probability']:.2%}"
                )

        high_risk_df = result["prediction_result"]

        high_risk_df = high_risk_df[
            high_risk_df["risk_level"] == "High Risk"
        ]

        st.subheader("🔴 High-Risk Patients")

        display_columns = [
            "patient_nbr",
            "age",
            "time_in_hospital",
            "num_medications",
            "readmission_probability"
        ]

        available_columns = [
            col for col in display_columns
            if col in high_risk_df.columns
        ]

        st.dataframe(
            high_risk_df[available_columns],
            use_container_width=True
        )
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == "Analytic Report":

    st.header("📈 Analytic Report")

    if "analytics_df" not in st.session_state:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state["analytics_df"]

        metrics = get_summary_metrics(df)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Uploaded Cases",
                metrics["uploaded_cases"]
            )

        with col2:
            st.metric(
                "High-Risk Patients",
                metrics["high_risk_patients"]
            )

        with col3:
            st.metric(
                "Average Risk Score",
                f"{metrics['average_risk_score']:.2%}"
            )

        with col4:
            st.metric(
                "Average Stay",
                f"{metrics['average_stay']:.1f} days"
            )

        with col5:
            st.metric(
                "Avg Medications",
                f"{metrics['average_medications']:.1f}"
            )

        st.divider()

        st.subheader("👥 Demographic Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                get_age_distribution_chart(df),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                get_gender_distribution_chart(df),
                use_container_width=True
            )

        st.plotly_chart(
            get_race_distribution_chart(df),
            use_container_width=True
        )

        st.divider()

        st.subheader("🏥 Clinical Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                get_hospital_stay_chart(df),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                get_medication_chart(df),
                use_container_width=True
            )

        st.divider()

        st.subheader("🎯 Readmission Risk Analysis")

        st.plotly_chart(
            get_risk_distribution_chart(df),
            use_container_width=True
        )

        st.divider()

        st.subheader("🧩 Patient Segmentation")

        st.plotly_chart(
            get_cluster_heatmap(df),
            use_container_width=True
        )

        st.plotly_chart(
            get_cluster_risk_chart(df),
            use_container_width=True
        )

        st.divider()

        st.subheader("🤖 AI-Generated Insights")

        insights = generate_insights_input(df)

        st.write(insights)

        st.markdown(f"""
        - The uploaded dataset contains **{insights['total_cases']:,}** patient records.

        - **{insights['high_risk_rate']:.1f}%** of patients are predicted to be at high risk of readmission.

        - Patients stay in hospital for an average of **{insights['average_stay']:.1f} days**.

        - Patients receive an average of **{insights['average_medications']:.1f} medications**.

        - **Cluster {insights['highest_risk_cluster']}** exhibits the highest predicted readmission risk (**{insights['highest_risk_cluster_rate']:.1f}%**).
        """)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == "Natural Language Analysis":

    st.header("Natural Language Query")
    st.caption(
        "Ask questions about the medical dataset using natural language."
    )

    # User input
    question = st.chat_input(
        "Example: How many patient was readmitted?"
    )

    clear_chat = st.button("🗑️ Clear Chat")

    if clear_chat:
        st.session_state.chat_history = []

    if question:

        with st.spinner("Processing query..."):

            result = handle_query(question)

            st.session_state.chat_history.append({
                "question": question,
                "result": result
            })

    st.divider()

    for chat in reversed(st.session_state.chat_history):

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):

            result = chat["result"]

            if isinstance(result, str):
                st.error(result)

            elif isinstance(result, dict):
                st.success("Prediction Result")
                st.json(result)

            elif isinstance(result, plt.Figure):
                st.pyplot(
                    result,
                    use_container_width=False
                    )

            elif isinstance(result, pd.DataFrame):
                st.dataframe(result)

            else:
                st.write(result)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

