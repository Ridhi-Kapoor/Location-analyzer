import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import polyline
from scipy.stats import ttest_ind

# ------------------------
# CONFIG & STYLING
# ------------------------
st.set_page_config(
    page_title="Location Analyzer",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e4e4f5;
}

/* ---------- APP BACKGROUND ---------- */
.stApp {
    background: linear-gradient(180deg, #0b0b13 0%, #111122 100%);
}

/* ---------- CONTAINER SPACING ---------- */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ---------- HEADERS ---------- */
.main-header {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.sub-header {
    font-size: 1rem;
    color: #9ca3af;
    margin-bottom: 2rem;
}

/* ---------- SECTION ---------- */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #c084fc;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-left: 4px solid #a855f7;
    padding-left: 10px;
}

/* ---------- METRIC CARDS ---------- */
.metric-card {
    background: linear-gradient(135deg, #7c3aed, #4c1d95);
    padding: 1.6rem;
    border-radius: 14px;
    color: white;
    text-align: center;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.3);
    transition: 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
}

.metric-card-neutral {
    background: #1a1a2e;
    padding: 1.6rem;
    border-radius: 14px;
    border: 1px solid #2e2e48;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
}

.metric-label {
    font-size: 0.75rem;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ---------- STAT CARDS (ANALYTICS) ---------- */
.stat-card {
    background: #141427;
    padding: 1.4rem;
    border-radius: 12px;
    border: 1px solid #2e2e48;
    box-shadow: 0 4px 14px rgba(0,0,0,0.4);
}

/* ---------- INFO BANNER ---------- */
.info-banner {
    background: linear-gradient(90deg, #4c1d95, #7c3aed);
    padding: 1.2rem;
    border-radius: 10px;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

.info-banner p {
    margin: 0;
    color: #ede9fe;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #2e2e48;
}

/* ---------- SLIDERS ---------- */
.stSlider > div > div {
    color: #a855f7;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    background: #141427;
    padding: 6px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    color: #9ca3af;
    padding: 10px 18px;
    border-radius: 8px;
}

.stTabs [aria-selected="true"] {
    background: #7c3aed;
    color: white;
}

/* ---------- DATAFRAME ---------- */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- SPACING FIXES ---------- */
.stMarkdown {
    margin-bottom: 0.5rem;
}

.stPlotlyChart, .stPyplot {
    margin-top: 1rem;
    margin-bottom: 2rem;
}

/* ---------- REMOVE FOOTER ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

API_KEY = "YOUR_API_KEY"

# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("**Weight Parameters**")
    st.caption("Adjust how each factor influences the location score")
    
    w_pop = st.slider("Population Density", 0.0, 1.0, 0.3, help="Higher weight = prioritize populated areas")
    w_traffic = st.slider("Foot Traffic", 0.0, 1.0, 0.3, help="Higher weight = prioritize high-traffic zones")
    w_rent = st.slider("Rental Cost", 0.0, 1.0, 0.2, help="Higher weight = penalize expensive areas more")
    w_comp = st.slider("Competition", 0.0, 1.0, 0.2, help="Higher weight = avoid saturated markets")
    
    st.markdown("---")
    
    # Validate weights
    total_weight = w_pop + w_traffic + w_rent + w_comp
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_weight:.2f}. Consider normalizing to 1.0 for balanced scoring.")

# ------------------------
# LOAD DATA
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/zomato.csv", encoding="latin1", on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Latitude": "lat",
        "Longitude": "lon",
        "Aggregate rating": "rating",
        "Votes": "votes",
        "Average Cost for two": "cost"
    })
    return df

df = load_data()

# ------------------------
# HEADER
# ------------------------
st.markdown('<p class="main-header">Location Performance Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data-driven site selection for restaurant ventures</p>', unsafe_allow_html=True)

# City selector in main area for prominence
col_city, col_spacer = st.columns([1, 3])
with col_city:
    city = st.selectbox("Select City", df["City"].dropna().unique(), label_visibility="collapsed")

df = df[df["City"] == city].copy()
df = df.dropna(subset=["lat", "lon"])

# ------------------------
# DATA WRANGLING
# ------------------------
df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

df["votes"].fillna(df["votes"].median(), inplace=True)
df["rating"].fillna(df["rating"].median(), inplace=True)
df["cost"].fillna(df["cost"].median(), inplace=True)

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

for col in ["votes", "cost"]:
    df = remove_outliers(df, col)

df["population"] = np.log1p(df["votes"])
df["traffic"] = df["rating"] * 20
df["rent"] = np.log1p(df["cost"])
df["competition"] = df.groupby("Locality")["Locality"].transform("count")
df["competition"] = np.log1p(df["competition"])

def norm(x):
    return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0

df["pop_n"] = norm(df["population"])
df["traffic_n"] = norm(df["traffic"])
df["rent_n"] = norm(df["rent"])
df["comp_n"] = norm(df["competition"])

df["score"] = (
    w_pop * df["pop_n"] +
    w_traffic * df["traffic_n"] -
    w_rent * df["rent_n"] -
    w_comp * df["comp_n"]
)

# ------------------------
# CLUSTERING
# ------------------------
features = df[["lat", "lon", "population", "traffic", "rent", "competition"]]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=min(10, len(df)), random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(features_scaled)

cluster_df = df.groupby("cluster").agg({
    "lat": "mean", "lon": "mean",
    "population": "mean", "traffic": "mean",
    "rent": "mean", "competition": "mean",
    "score": "mean"
}).reset_index()

best_cluster = cluster_df.loc[cluster_df["score"].idxmax()]

# ------------------------
# ROUTE FUNCTION
# ------------------------
def get_routes(start_lat, start_lon, end_lat, end_lon):
    url = "[maps.googleapis.com](https://maps.googleapis.com/maps/api/directions/json)"
    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{end_lat},{end_lon}",
        "alternatives": "true",
        "departure_time": "now",
        "key": API_KEY
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        routes = []
        if res.get("status") == "OK":
            for r in res["routes"]:
                points = polyline.decode(r["overview_polyline"]["points"])
                leg = r["legs"][0]
                routes.append((points, leg["duration"]["text"], leg["distance"]["text"]))
        return routes
    except:
        return []

# ------------------------
# TABS
# ------------------------
tab1, tab2 = st.tabs(["📍 Location Map", "📈 Analytics"])

# ------------------------
# MAP TAB
# ------------------------
with tab1:
    st.markdown('<p class="section-header">Recommended Location</p>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_cluster['lat']:.4f}</div>
            <div class="metric-label">Latitude</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_cluster['lon']:.4f}</div>
            <div class="metric-label">Longitude</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_cluster['score']:.3f}</div>
            <div class="metric-label">Location Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Data Points</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Map
    m = folium.Map(
        location=[df["lat"].mean(), df["lon"].mean()],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    df["heat_weight"] = df["population"] * 0.6 + df["traffic"] * 0.4
    df["heat_weight"] = np.log1p(df["heat_weight"])
    df["heat_weight"] = norm(df["heat_weight"])
    
    heat_data = [[row["lat"], row["lon"], row["heat_weight"]] for _, row in df.iterrows()]
    
    HeatMap(
        heat_data,
        radius=25,
        blur=35,
        min_opacity=0.2,
        gradient={0.0: "#3b82f6", 0.25: "#06b6d4", 0.5: "#10b981", 0.75: "#f59e0b", 1.0: "#ef4444"}
    ).add_to(m)
    
    folium.Marker(
        [best_cluster["lat"], best_cluster["lon"]],
        popup=folium.Popup(
            f"<b>Recommended Site</b><br>Score: {best_cluster['score']:.3f}",
            max_width=200
        ),
        icon=folium.Icon(color="red", icon="star", prefix="fa")
    ).add_to(m)
    
    folium_static(m, width=1100, height=500)
    
    # Navigation section
    st.markdown('<p class="section-header">Navigation</p>', unsafe_allow_html=True)
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 2])
    
    with col_nav1:
        user_lat = st.number_input("Your Latitude", value=28.6, format="%.4f")
    
    with col_nav2:
        user_lon = st.number_input("Your Longitude", value=77.2, format="%.4f")
    
    with col_nav3:
        st.markdown("<br>", unsafe_allow_html=True)
        maps_url = f"[google.com](https://www.google.com/maps/dir/{user_lat},{user_lon}/{best_cluster['lat']},{best_cluster['lon']}"


        st.markdown(f"[🗺️ Open in Google Maps]({maps_url})")

# ------------------------
# ANALYTICS TAB
# ------------------------
with tab2:
    # Summary stats
    st.markdown('<p class="section-header">Cluster Performance</p>', unsafe_allow_html=True)
    
    styled_cluster_df = cluster_df.sort_values("score", ascending=False).round(3)
    styled_cluster_df.columns = ["Cluster", "Lat", "Lon", "Population", "Traffic", "Rent", "Competition", "Score"]
    st.dataframe(
        styled_cluster_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Hypothesis Testing
    st.markdown('<p class="section-header">Statistical Analysis</p>', unsafe_allow_html=True)
    
    high = df[df["traffic"] > df["traffic"].median()].copy()
    low = df[df["traffic"] <= df["traffic"].median()].copy()
    
    t_stat, p_val = ttest_ind(high["score"], low["score"])
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-box-title">T-Statistic</div>
            <div class="stat-box-value">{t_stat:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-box-title">P-Value</div>
            <div class="stat-box-value">{p_val:.5f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        significance = "Significant" if p_val < 0.05 else "Not Significant"
        color = "#10b981" if p_val < 0.05 else "#6b7280"
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-box-title">Result (α=0.05)</div>
            <div class="stat-box-value" style="color: {color};">{significance}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-banner">
        <p><strong>Hypothesis:</strong> High-traffic areas yield significantly different location scores than low-traffic areas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown('<p class="section-header">Distributions</p>', unsafe_allow_html=True)
    
    # Set consistent style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#667eea', '#764ba2', '#f59e0b', '#10b981']
    
    high["group"] = "High Traffic"
    low["group"] = "Low Traffic"
    combined = pd.concat([high, low])
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x="group", y="score", data=combined, ax=ax,
            palette=[colors[0], colors[1]],
            width=0.5
        )
        ax.set_xlabel("")
        ax.set_ylabel("Location Score", fontsize=10)
        ax.set_title("Score by Traffic Level", fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_viz2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(high["score"], fill=True, label="High Traffic", ax=ax, color=colors[0], alpha=0.6)
        sns.kdeplot(low["score"], fill=True, label="Low Traffic", ax=ax, color=colors[1], alpha=0.6)
        ax.set_xlabel("Location Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Score Density Distribution", fontsize=12, fontweight='bold')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    
    col_viz3, col_viz4 = st.columns(2)
    
    with col_viz3:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["score"], bins=25, color=colors[0], edgecolor='white', alpha=0.8)
        ax.set_xlabel("Location Score", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("Score Distribution", fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_viz4:
        corr_cols = ["population", "traffic", "rent", "competition", "score"]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            df[corr_cols].corr(),
            annot=True,
            cmap="RdYlBu_r",
            ax=ax,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5
        )
        ax.set_title("Feature Correlations", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
