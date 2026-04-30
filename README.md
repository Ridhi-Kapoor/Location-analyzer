# 📍 Location Performance Analyzer

A data-driven site selection tool designed to help entrepreneurs and restaurant owners identify optimal locations for new ventures. This application leverages the Zomato dataset to provide actionable insights through advanced data analytics, machine learning, and interactive visualizations.

## 🚀 Overview

The **Location Performance Analyzer** evaluates potential business sites based on a multi-factor scoring model. By analyzing historical restaurant data, the tool identifies high-potential "hotspots" using clustering algorithms and validates findings with statistical hypothesis testing.

### Key Features

- **Interactive Dashboard:** Built with Streamlit for a seamless user experience.
- **Customizable Scoring:** Adjust weights for Population Density, Foot Traffic, Rental Cost, and Competition to align with your business strategy.
- **Geospatial Visualization:** Interactive Folium maps with heatmaps showing density and recommended site markers.
- **Clustering Analysis:** Uses K-Means clustering to group similar locations and identify the best-performing clusters.
- **Statistical Validation:** Performs T-tests to verify the significance of factors like foot traffic on location scores.
- **Navigation Integration:** Direct links to Google Maps for recommended site navigation.

## 🛠️ Tech Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (K-Means, MinMaxScaler)
- **Visualization:** [Folium](https://python-visualization.github.io/folium/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/python/)
- **Statistics:** [SciPy](https://www.scipy.org/)
- **Geospatial Tools:** [Geopy](https://geopy.readthedocs.io/), [Polyline](https://pypi.org/project/polyline/)

## 📋 Prerequisites

- Python 3.8 or higher
- A Google Maps API Key (optional, for advanced navigation features)

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/location-analyzer.git
   cd location-analyzer
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **API Configuration (Optional):**
   Open `location.py` and replace `"YOUR_API_KEY"` with your actual Google Maps API key to enable full navigation routing features.

## 🏃 Running the Application

To launch the Streamlit dashboard, run:

```bash
streamlit run location.py
```

The application will be available in your browser at `http://localhost:8501`.

## 📊 Data Source

The project uses a Zomato dataset (located in `data/zomato.csv`) containing restaurant details, locations (latitude/longitude), ratings, and costs across various cities.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
