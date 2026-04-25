import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import polyline

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(layout="wide")
st.title("Location Performance Analyzer")
st.caption("AI + GIS Based Restaurant Location Recommendation")

API_KEY = "YOUR_API_KEY"

# ------------------------
# SIDEBAR
# ------------------------
st.sidebar.header("Configuration")

w_pop = st.sidebar.slider("Population", 0.0, 1.0, 0.3)
w_traffic = st.sidebar.slider("Traffic", 0.0, 1.0, 0.3)
w_rent = st.sidebar.slider("Rent", 0.0, 1.0, 0.2)
w_comp = st.sidebar.slider("Competition", 0.0, 1.0, 0.2)

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
# CITY FILTER
# ------------------------
city = st.sidebar.selectbox("Select City", df["City"].dropna().unique())
df = df[df["City"] == city]
df = df.dropna(subset=["lat", "lon"])

# ------------------------
# FEATURES
# ------------------------
df["population"] = df.get("votes", 0)
df["traffic"] = df.get("rating", 0) * 20
df["rent"] = df.get("cost", 10000)
df["competition"] = df.groupby("Locality")["Locality"].transform("count")

def norm(x):
    if x.max() == x.min():
        return pd.Series(0, index=x.index)
    return (x - x.min()) / (x.max() - x.min())

# ------------------------
# CLUSTERING
# ------------------------
features = df[["lat", "lon", "population", "traffic", "rent", "competition"]]

# Normalize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=min(10, len(df)), random_state=42)
df["cluster"] = kmeans.fit_predict(features_scaled)

cluster_df = df.groupby("cluster").agg({
    "lat": "mean",
    "lon": "mean",
    "population": "mean",
    "traffic": "mean",
    "rent": "mean",
    "competition": "mean"
}).reset_index()

for col in ["population","traffic","rent","competition"]:
    cluster_df[col+"_n"] = norm(cluster_df[col])

cluster_df["score"] = (
    w_pop * cluster_df["population_n"] +
    w_traffic * cluster_df["traffic_n"] -
    w_rent * cluster_df["rent_n"] -
    w_comp * cluster_df["competition_n"]
)

best_cluster = cluster_df.loc[cluster_df["score"].idxmax()]

# ------------------------
# ROUTE FUNCTION
# ------------------------
def get_routes(start_lat, start_lon, end_lat, end_lon):
    url = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{end_lat},{end_lon}",
        "alternatives": "true",
        "departure_time": "now",
        "key": API_KEY
    }

    res = requests.get(url, params=params).json()

    routes = []
    if res.get("status") == "OK":
        for r in res["routes"]:
            points = polyline.decode(r["overview_polyline"]["points"])
            leg = r["legs"][0]
            duration = leg["duration"]["text"]
            distance = leg["distance"]["text"]
            routes.append((points, duration, distance))
    return routes

# ------------------------
# TABS
# ------------------------
tab1, tab2 = st.tabs(["🗺️ Map", "📊 Analytics"])

# ------------------------
# MAP TAB
# ------------------------
with tab1:
    st.subheader(f"Recommended Location - {city}")

    # Coordinates
    col1, col2, col3 = st.columns(3)
    col1.metric("Latitude", round(best_cluster["lat"], 5))
    col2.metric("Longitude", round(best_cluster["lon"], 5))
    col3.metric("Score", round(best_cluster["score"], 3))

    # Map
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=12)

    # ------------------------
    # UPDATED HEATMAP (ONLY CHANGE)
    # ------------------------
    df["heat_weight"] = df["population"] * 0.6 + df["traffic"] * 0.4

    # LOG scaling for better spread (fixes patch issue)
    df["heat_weight"] = np.log1p(df["heat_weight"])

    # Normalize
    df["heat_weight"] = (df["heat_weight"] - df["heat_weight"].min()) / (
        df["heat_weight"].max() - df["heat_weight"].min()
    )

    heat_data = [
        [row["lat"], row["lon"], row["heat_weight"]]
        for _, row in df.iterrows()
    ]

    HeatMap(
        heat_data,
        radius=25,
        blur=35,
        min_opacity=0.2,
        max_zoom=13,
        gradient={
            0.2: "blue",
            0.4: "cyan",
            0.6: "lime",
            0.8: "yellow",
            1.0: "red"
        }
    ).add_to(m)

    # ------------------------
    # HEATMAP LEGEND
    # ------------------------
    heatmap_legend = '''
    <div style="
    position: fixed;
    bottom: 50px; right: 50px;
    width: 260px; height: 130px;
    background-color: white;
    border: 2px solid grey;
    z-index: 9999;
    font-size: 14px;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
    <b>Heatmap (Demand Intensity)</b><br><br>
    <div style="display: flex; align-items: center;">
    <span style="margin-right: 6px;">Low</span>
    <div style="
    width: 150px; height: 12px;
    background: linear-gradient(to right,
    blue, cyan, lime, yellow, red);
    border:1px solid black;">
    </div>
    <span style="margin-left: 6px;">High</span>
    </div>
    <br>
    <small>Based on population & traffic</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(heatmap_legend))

    # Points
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="black",
            fill=True
        ).add_to(m)

    # Best location
    folium.Marker(
        [best_cluster["lat"], best_cluster["lon"]],
        popup="Recommended Location",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    # ROUTE
    st.subheader("Navigation")

    user_lat = st.number_input("Your Latitude", value=28.6)
    user_lon = st.number_input("Your Longitude", value=77.2)

    routes = get_routes(user_lat, user_lon, best_cluster["lat"], best_cluster["lon"])

    if routes:
        routes = sorted(routes, key=lambda x: int(x[1].split()[0]))

        for i, r in enumerate(routes):
            points, duration, distance = r

            if i == 0:
                st.success(f"Best Route → {distance}, {duration}")
                folium.PolyLine(points, color="blue", weight=6).add_to(m)
            else:
                st.info(f"Alternative Route {i+1} → {distance}, {duration}")
                folium.PolyLine(points, color="gray", weight=3).add_to(m)

    route_link = f"https://www.google.com/maps/dir/{user_lat},{user_lon}/{best_cluster['lat']},{best_cluster['lon']}"
    st.markdown(f"[Open in Google Maps]({route_link})")

    folium_static(m)

# ------------------------
# ANALYTICS TAB
# ------------------------
with tab2:
    st.subheader("Cluster Analysis")

    st.dataframe(cluster_df.sort_values("score", ascending=False))

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", round(cluster_df["score"].mean(),3))
    col2.metric("Std Dev", round(cluster_df["score"].std(),3))
    col3.metric("Variance", round(cluster_df["score"].var(),3))

    fig, ax = plt.subplots()
    ax.hist(cluster_df["score"], bins=10)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.boxplot(cluster_df["score"])
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(cluster_df.corr(), annot=True)
    st.pyplot(fig)