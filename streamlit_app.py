# streamlit_app.py
import streamlit as st
import plotly.express as px
import requests
import pandas as pd
import os

st.set_page_config(page_title="BayBot", layout="wide")
st.title("🌊 BayBot — Water Quality Dashboard")
st.caption("Real-time water quality monitoring powered by Flask, MongoDB, and Hugging Face RAG.")

# ---- DETECT ENVIRONMENT ----
# When running locally, use Flask API
# When deployed on Streamlit Cloud, call database directly
IS_LOCAL = os.getenv("IS_LOCAL", "false") == "true"
API_URL = "http://localhost:5000" if IS_LOCAL else "https://rag-baybot.onrender.com"

# ---- FETCH SENSOR DATA ----
try:
    if IS_LOCAL:
        sensor_response = requests.get(f"{API_URL}/api/sensors?limit=500")
        data = sensor_response.json()
    else:
        from database import get_collection
        collection = get_collection()
        data = list(collection.find({}, {"_id": 0, "embedding": 0}).limit(500))
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M")
    data_loaded = True
except Exception as e:
    df = pd.DataFrame()
    data_loaded = False
    st.error(f"⚠️ Could not load sensor data: {e}")
    st.stop()

# ---- SIDEBAR FILTERS ----
st.sidebar.header("🔧 Filters")
st.sidebar.caption("Filters affect charts and filtered stats only.")

start_date = st.sidebar.date_input("Start Date", value=df["date"].min().date())
end_date = st.sidebar.date_input("End Date", value=df["date"].max().date())

locations = ["All"] + sorted(df["location"].unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations)

parameter = st.sidebar.selectbox(
    "Parameter to Visualize",
    ["temperature", "salinity", "dissolved_oxygen"],
    format_func=lambda x: {
        "temperature": "Temperature (°C)",
        "salinity": "Salinity (ppt)",
        "dissolved_oxygen": "Dissolved Oxygen (mg/L)"
    }[x]
)

# ---- APPLY FILTERS ----
filtered_df = df[
    (df["date"] >= pd.Timestamp(start_date)) &
    (df["date"] <= pd.Timestamp(end_date))
]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df["location"] == selected_location]

# ---- GLOBAL STATS ----
st.subheader("📊 Dataset Overview — All Records")
st.caption("These figures represent the full dataset regardless of filters.")

try:
    if IS_LOCAL:
        stats_response = requests.get(f"{API_URL}/api/stats")
        stats = stats_response.json()
    else:
        from database import get_global_stats
        stats = get_global_stats()
    stats_loaded = True
except Exception:
    stats = {}
    stats_loaded = False

if stats_loaded:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌡️ Avg Temperature", f"{stats.get('avg_temperature', 'N/A')}°C",
                delta=f"Max {stats.get('max_temperature', 'N/A')}°C", delta_color="off")
    col2.metric("🧂 Avg Salinity", f"{stats.get('avg_salinity', 'N/A')} ppt")
    col3.metric("💧 Avg Dissolved Oxygen", f"{stats.get('avg_dissolved_oxygen', 'N/A')} mg/L")
    col4.metric("🌡️ Temp Range", f"{stats.get('min_temperature', 'N/A')}°C – {stats.get('max_temperature', 'N/A')}°C")
    col5.metric("📁 Total Records", f"{stats.get('total_records', 0):,}")
else:
    st.warning("⚠️ Could not load global statistics.")

st.divider()

# ---- FILTERED STATS ----
st.subheader("🔍 Filtered Selection Stats")
st.caption("These figures update based on your sidebar filters.")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
f_col1.metric("Avg Temperature (filtered)", f"{filtered_df['temperature'].mean():.2f}°C")
f_col2.metric("Avg Salinity (filtered)", f"{filtered_df['salinity'].mean():.2f} ppt")
f_col3.metric("Avg DO (filtered)", f"{filtered_df['dissolved_oxygen'].mean():.2f} mg/L")
f_col4.metric("Records in Selection", f"{len(filtered_df):,}")

st.divider()

# ---- MAIN DYNAMIC CHART ----
st.subheader(f"📈 {parameter.replace('_', ' ').title()} Over Time")

fig_main = px.line(
    filtered_df,
    x="date",
    y=parameter,
    color="location",
    title=f"{parameter.replace('_', ' ').title()} Over Time by Location",
    labels={
        "date": "Date",
        parameter: {
            "temperature": "Temperature (°C)",
            "salinity": "Salinity (ppt)",
            "dissolved_oxygen": "Dissolved Oxygen (mg/L)"
        }[parameter],
        "location": "Location"
    }
)
fig_main.update_layout(plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig_main, use_container_width=True)

st.divider()

# ---- ALL THREE CHARTS ----
st.subheader("📊 All Parameters Overview")

col_a, col_b, col_c = st.columns(3)

with col_a:
    fig_temp = px.line(filtered_df, x="date", y="temperature", color="location",
                       title="Temperature (°C)", labels={"date": "Date", "temperature": "°C"})
    fig_temp.update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig_temp, use_container_width=True)

with col_b:
    fig_sal = px.line(filtered_df, x="date", y="salinity", color="location",
                      title="Salinity (ppt)", labels={"date": "Date", "salinity": "ppt"})
    fig_sal.update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig_sal, use_container_width=True)

with col_c:
    fig_do = px.line(filtered_df, x="date", y="dissolved_oxygen", color="location",
                     title="Dissolved Oxygen (mg/L)", labels={"date": "Date", "dissolved_oxygen": "mg/L"})
    fig_do.update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig_do, use_container_width=True)

st.divider()

# ---- CORRELATION SCATTER ----
st.subheader("🔍 Parameter Correlation")

x_axis = st.selectbox("X Axis", ["temperature", "salinity", "dissolved_oxygen"],
                      format_func=lambda x: x.replace("_", " ").title())
y_axis = st.selectbox("Y Axis", ["dissolved_oxygen", "temperature", "salinity"],
                      format_func=lambda x: x.replace("_", " ").title())

fig_scatter = px.scatter(filtered_df, x=x_axis, y=y_axis, color="location",
                         opacity=0.6, trendline="ols",
                         title=f"{x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}")
fig_scatter.update_layout(plot_bgcolor="white")
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ---- RAG CHATBOT ----
st.header("🤖 Ask BayBot a Question")
st.caption(
    "BayBot answers using real sensor records retrieved from MongoDB via vector search. "
    "Source records and relevance scores are shown below every answer."
)

question = st.text_input("Your question",
                         placeholder="e.g. What was the average temperature in March at North Bay?")

if st.button("Ask BayBot", type="primary"):
    if question:
        with st.spinner("Retrieving relevant sensor records and generating answer..."):
            try:
                if IS_LOCAL:
                    response = requests.post(
                        f"{API_URL}/api/ask",
                        json={"question": question},
                        timeout=120
                    )
                    result = response.json()
                else:
                    from rag import answer as rag_answer
                    result = rag_answer(question)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if result.get("warning"):
            st.warning(f"⚠️ {result['warning']}")

        st.success(result["answer"])

        with st.expander("📄 View Source Records & Relevance Scores"):
            sources_df = pd.DataFrame(result["sources"])
            if "score" in sources_df.columns:
                sources_df = sources_df.rename(columns={"score": "relevance_score"})
                sources_df = sources_df.sort_values("relevance_score", ascending=False)
            if "description" in sources_df.columns:
                sources_df = sources_df.drop(columns=["description"])
            st.dataframe(sources_df, use_container_width=True)
    else:
        st.warning("Please enter a question before clicking Ask.")