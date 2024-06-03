import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add logo to the top right corner
logo_url = "logo.png"  # Replace with your logo URL
st.sidebar.image(logo_url, use_column_width=True)

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('predictive_maintenance.csv')  # Assuming the file is in the same directory as the script
    return data

data = load_data()

# Preprocessing
# For the sake of example, let's use some columns as features
X = data[['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]']]
# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Failure Type'])

# Rename columns to remove special characters
X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in X.columns]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create model and evaluate its performance
def evaluate_model(model):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_cv_score = np.mean(cv_scores)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, f1, mean_cv_score, conf_matrix, y_pred

# Function to display evaluation results
def show_results(accuracy, f1, mean_cv_score, conf_matrix, predicted_failure, repair_status, repair_action):
    st.write("Evaluation Results:")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"F1 Score: {f1}")
    st.write(f"Cross-Validation Score: {mean_cv_score}")
    
    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)
    
    # Show predicted failure and repair status
    st.write("Predicted Failure, Repair Status, and Repair Action:")
    results_df = pd.DataFrame({
        'Predicted Failure': predicted_failure,
        'Repair Status': repair_status,
        'Repair Action': repair_action
    })
    st.write(results_df)

# Function to get status for a specific item number
def get_status_for_item(item_number, model, X_test):
    if item_number in X_test.index:
        prediction = model.predict([X_test.loc[item_number]])[0]
        failure_types = [
            "Tidak Ada Kegagalan",
            "Kegagalan Pembuangan Panas",
            "Kegagalan Daya",
            "Kegagalan Overstrain",
            "Kegagalan Keausan Alat"
        ]
        failure_type = failure_types[prediction] if prediction < len(failure_types) else "Kegagalan Tidak Dikenal"
        repair_status = "Tidak Butuh Perbaikan" if failure_type == "Tidak Ada Kegagalan" else "Butuh Perbaikan"
        repair_action = "Santai Saja" if repair_status == "Tidak Butuh Perbaikan" else "Perbaiki Segera"
        return failure_type, repair_status, repair_action
    else:
        return None, None, None

# Streamlit app
st.title("PREDIKSI AKURASI DAN KEGAGALAN")
st.success("PILIH MODEL YANG TELAH DISEDIAKAN PROGRAM")
# Sidebar for Model Selection
st.sidebar.title("Pemilihan Model")
model_name = st.sidebar.selectbox("Pilih Model", ["Random Forest", "Gradient Boosting", "SVM Classifier", "Neural Networks"])

# Button for each model
button_clicked = st.sidebar.button("Evaluasi")

# Checkbox for showing/hiding histogram
show_histogram = st.sidebar.checkbox("Tampilkan Histogram")

if button_clicked:
    if model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier()
    elif model_name == "SVM Classifier":
        model = SVC()
    elif model_name == "Neural Networks":
        model = MLPClassifier()

    accuracy, f1, mean_cv_score, conf_matrix, y_pred = evaluate_model(model)

    # Map predicted labels to failure types and determine repair status and action
    failure_types = [
        "Tidak Ada Kegagalan",
        "Kegagalan Pembuangan Panas",
        "Kegagalan Daya",
        "Kegagalan Overstrain",
        "Kegagalan Keausan Alat"
    ]
    predicted_failure = []
    repair_status = []
    repair_action = []
    for label in y_pred:
        if label < len(failure_types):
            predicted_failure.append(failure_types[label])
            if failure_types[label] == "Tidak Ada Kegagalan":
                repair_status.append("Tidak Butuh Perbaikan")
                repair_action.append("Santai Saja")
            else:
                repair_status.append("Butuh Perbaikan")
                repair_action.append("Perbaiki Segera")
        else:
            predicted_failure.append("Kegagalan Tidak Dikenal")
            repair_status.append("Butuh Perbaikan")
            repair_action.append("Perbaiki Segera")

    show_results(accuracy, f1, mean_cv_score, conf_matrix, predicted_failure, repair_status, repair_action)

    st.balloons()

    if show_histogram:
        # Plot histogram for failure types, repair status, and repair action
        st.subheader("Histogram for Failure Types, Repair Status, and Repair Action")
        plt.figure(figsize=(12, 8))

        # Count occurrences of each failure type
        failure_counts = {ft: predicted_failure.count(ft) for ft in failure_types}
        repair_status_counts = {rs: repair_status.count(rs) for rs in set(repair_status)}
        repair_action_counts = {ra: repair_action.count(ra) for ra in set(repair_action)}

        # Plot histograms
        plt.subplot(3, 1, 1)
        plt.bar(failure_counts.keys(), failure_counts.values(), color='skyblue')
        plt.xlabel('Failure Types')
        plt.ylabel('Frequency')

        plt.subplot(3, 1, 2)
        plt.bar(repair_status_counts.keys(), repair_status_counts.values(), color='salmon')
        plt.xlabel('Repair Status')
        plt.ylabel('Frequency')

        plt.subplot(3, 1, 3)
        plt.bar(repair_action_counts.keys(), repair_action_counts.values(), color='lightgreen')
        plt.xlabel('Repair Action')
        plt.ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(plt)
