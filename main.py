import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add logo to the top right corner
logo_url = "logo.png"  # Replace with your logo URL
st.sidebar.image(logo_url, use_column_width=True)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('predictive_maintenance.csv')  # Assuming the file is in the same directory as the script
    return data

# Load data
data = load_data()

# Sidebar
st.sidebar.title("Visualisasi Data")
visualization_type = st.sidebar.selectbox("Pilih Tipe Visualisasi", ["Pie Chart", "Scatter Plot", "Simple Area Chart", "Simple Bar Chart", "Simple Line Chart"])

# Function to preprocess data if necessary
def preprocess_data(data):
    # For simplicity, let's just use a subset of columns
    processed_data = data[['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]', 'Failure Type']]
    return processed_data

# Preprocess data
processed_data = preprocess_data(data)

# Visualizations
if visualization_type == "Pie Chart":
    st.subheader("Pie Chart")
    fig, ax = plt.subplots()
    data['Failure Type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

elif visualization_type == "Scatter Plot":
    st.subheader("Scatter Plot")
    x_variable = st.selectbox("Pilih Variabel X", processed_data.columns[:-1])
    y_variable = st.selectbox("Pilih Variabel Y", processed_data.columns[:-1])
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_variable, y=y_variable, hue='Failure Type', data=processed_data, ax=ax)
    st.pyplot(fig)

elif visualization_type == "Simple Area Chart":
    st.subheader("Simple Area Chart")
    x_variable = st.selectbox("Pilih Variabel X", processed_data.columns[:-1])
    y_variable = st.selectbox("Pilih Variabel Y", processed_data.columns[:-1])
    fig, ax = plt.subplots()
    sns.lineplot(x=x_variable, y=y_variable, hue='Failure Type', data=processed_data, ax=ax)
    st.pyplot(fig)

elif visualization_type == "Simple Bar Chart":
    st.subheader("Simple Bar Chart")
    x_variable = st.selectbox("Pilih Variabel X", processed_data.columns[:-1])
    y_variable = st.selectbox("Pilih Variabel Y", processed_data.columns[:-1])
    fig, ax = plt.subplots()
    sns.barplot(x=x_variable, y=y_variable, hue='Failure Type', data=processed_data, ax=ax)
    st.pyplot(fig)

elif visualization_type == "Simple Line Chart":
    st.subheader("Simple Line Chart")
    x_variable = st.selectbox("Pilih Variabel X", processed_data.columns[:-1])
    y_variable = st.selectbox("Pilih Variabel Y", processed_data.columns[:-1])
    fig, ax = plt.subplots()
    sns.lineplot(x=x_variable, y=y_variable, hue='Failure Type', data=processed_data, ax=ax)
    st.pyplot(fig)
