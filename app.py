import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
import os

# Set page layout
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# App Title
st.title("💳 Credit Card Customer Segmentation")
st.markdown("This app uses **Hierarchical Clustering** to group customers based on their banking behavior.")

# --- STEP 1: DATA LOADING ---
DATA_FILENAME = "Credit Card Customer Data.csv"

def process_clustering(df):
    """Encapsulates the ML logic from your notebook"""
    # Preprocessing
    X = df.drop(['Sl_No', 'Customer Key'], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering (Agglomerative)
    hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
    y_clusters = hc.fit_predict(X_scaled)
    df['Cluster'] = y_clusters
    
    # Create a KNN Classifier to 'predict' new inputs
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_scaled, y_clusters)
    
    return df, scaler, knn, X

# Attempt to load data automatically if file exists on GitHub
df = None
if os.path.exists(DATA_FILENAME):
    df = pd.read_csv(DATA_FILENAME)
else:
    st.warning(f"⚠️ **{DATA_FILENAME}** not found in repository.")
    uploaded_file = st.file_uploader("Please upload the 'Credit Card Customer Data.csv' file to continue", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

# --- STEP 2: APP LOGIC ---
if df is not None:
    # Run the clustering logic
    df_clustered, scaler, model, features = process_clustering(df)
    
    # Create Sidebar for User Inputs
    st.sidebar.header("User Feature Input")
    def get_user_input():
        avg_limit = st.sidebar.number_input("Avg Credit Limit", 3000, 200000, 50000)
        total_cards = st.sidebar.slider("Total Credit Cards", 1, 10, 4)
        v_bank = st.sidebar.slider("Total Visits Bank", 0, 5, 2)
        v_online = st.sidebar.slider("Total Visits Online", 0, 15, 2)
        calls = st.sidebar.slider("Total Calls Made", 0, 10, 3)
        
        return pd.DataFrame({
            'Avg_Credit_Limit': [avg_limit],
            'Total_Credit_Cards': [total_cards],
            'Total_visits_bank': [v_bank],
            'Total_visits_online': [v_online],
            'Total_calls_made': [calls]
        })

    user_data = get_user_input()

    # Layout: Two Columns
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Classification Result")
        if st.button("Identify Customer Cluster"):
            # Scale input and predict using the KNN proxy
            user_scaled = scaler.transform(user_data)
            prediction = model.predict(user_scaled)[0]
            
            st.metric(label="Assigned Cluster", value=f"Cluster {prediction}")
            
            # Insights based on your Notebook's cluster means
            if prediction == 0:
                st.info("💡 **Profile:** Low Credit Limit, High Bank Visits. (Traditional Banking)")
            elif prediction == 1:
                st.success("💡 **Profile:** High Credit Limit, High Online Visits. (Digital Premium)")
            else:
                st.info("💡 **Profile:** Mid-range Credit Limit, Frequent Phone Support.")

    with col2:
        st.subheader("Cluster Distribution")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_clustered, x='Avg_Credit_Limit', y='Total_visits_online', 
                        hue='Cluster', palette='viridis', s=60, ax=ax)
        st.pyplot(fig)

    # Show data table at bottom
    if st.checkbox("Show dataset with cluster labels"):
        st.write(df_clustered.head(10))

else:
    st.info("Waiting for data file...")
    st.stop()
