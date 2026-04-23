import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Location Analysis", layout="wide")

# ------------------------
# CUSTOM CSS
# ------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    font-size: 40px;
    font-weight: 600;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# HEADER
# ------------------------
st.title("Data-Driven Location Analysis")
st.caption("Decision-support system for optimal business location selection")

# ------------------------
# SIDEBAR
# ------------------------
st.sidebar.title("Configuration")

with st.sidebar.expander("Location Settings", True):
    city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore"])
    business = st.selectbox("Business Type", ["Restaurant", "Retail", "Gym"])

with st.sidebar.expander("Weight Adjustment", True):
    w_pop = st.slider("Population", 0.0, 1.0, 0.3)
    w_traffic = st.slider("Traffic", 0.0, 1.0, 0.3)
    w_rent = st.slider("Rent", 0.0, 1.0, 0.2)
    w_comp = st.slider("Competition", 0.0, 1.0, 0.2)

# ------------------------
# DATA
# ------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "lat": np.random.uniform(28.4, 28.8, 100),
        "lon": np.random.uniform(77.0, 77.4, 100),
        "population": np.random.randint(1000, 10000, 100),
        "traffic": np.random.randint(10, 100, 100),
        "rent": np.random.randint(5000, 50000, 100),
        "competition": np.random.randint(1, 20, 100)
    })
    return df

df = load_data()

# ------------------------
# NORMALIZATION
# ------------------------
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

df["pop_n"] = normalize(df["population"])
df["traffic_n"] = normalize(df["traffic"])
df["rent_n"] = normalize(df["rent"])
df["comp_n"] = normalize(df["competition"])

# ------------------------
# SCORING
# ------------------------
df["score"] = (
    w_pop * df["pop_n"] +
    w_traffic * df["traffic_n"] -
    w_rent * df["rent_n"] -
    w_comp * df["comp_n"]
)

# ------------------------
# TABS
# ------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analysis", "Model", "Map"])

# ------------------------
# OVERVIEW
# ------------------------
with tab1:
    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
    <div class="metric-card">
        <h4>Average Population</h4>
        <h2>{int(df["population"].mean())}</h2>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card">
        <h4>Average Rent</h4>
        <h2>{int(df["rent"].mean())}</h2>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card">
        <h4>Average Traffic</h4>
        <h2>{int(df["traffic"].mean())}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview")
    st.dataframe(df, use_container_width=True)

# ------------------------
# EDA
# ------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df["population"])
        ax.set_title("Population Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.scatter(df["population"], df["rent"])
        ax.set_title("Population vs Rent")
        st.pyplot(fig)

# ------------------------
# ML MODEL
# ------------------------
with tab3:
    st.subheader("Clustering Insights")

    X = df[["pop_n", "traffic_n", "rent_n", "comp_n"]]

    model = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = model.fit_predict(X)

    st.markdown("Cluster Distribution")
    st.bar_chart(df["cluster"].value_counts())

# ------------------------
# MAP
# ------------------------
with tab4:
    st.subheader("Location Visualization")

    m = folium.Map(
        location=[df["lat"].mean(), df["lon"].mean()],
        zoom_start=11
    )

    for _, row in df.iterrows():
        color = "green" if row["score"] > df["score"].median() else "red"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color,
            fill=True
        ).add_to(m)

    folium_static(m)

# ------------------------
# BEST LOCATION
# ------------------------
st.markdown("---")
st.subheader("Best Location Recommendation")

best = df.loc[df["score"].idxmax()]

st.markdown(f"""
<div style="background-color:#1f2937;padding:20px;border-radius:10px;">
    <h3>Optimal Location Identified</h3>
    <p><b>Latitude:</b> {best['lat']}</p>
    <p><b>Longitude:</b> {best['lon']}</p>
    <p><b>Score:</b> {round(best['score'],3)}</p>
</div>
""", unsafe_allow_html=True)