import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (KFold, cross_val_score, train_test_split,
                                     ShuffleSplit, LeaveOneOut)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report, roc_auc_score, roc_curve, 
                             mean_squared_error, log_loss, accuracy_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import io

# Function to load the Heart Attack dataset
@st.cache_data
def load_heart_data(uploaded_file):
    names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
             'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 
             'sex', 'smoking', 'time', 'DEATH_EVENT']
    dataframe = pd.read_csv(uploaded_file, names=names, header=0)
    return dataframe

# Function to load and preprocess the Water Quality dataset
@st.cache_data
def load_water_data(uploaded_file):
    names = ['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine',
             'chromium', 'copper', 'fluoride', 'bacteria', 'viruses', 'lead',
             'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium',
             'selenium', 'silver', 'uranium', 'is_safe']
    dataframe = pd.read_csv(uploaded_file, names=names)
    
    dataframe.replace('#NUM!', pd.NA, inplace=True)
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

    # Impute missing values for numeric data (mean strategy)
    numeric_imputer = SimpleImputer(strategy='mean')
    dataframe.iloc[:, :-1] = numeric_imputer.fit_transform(dataframe.iloc[:, :-1])

    # Encode target variable 'is_safe' if necessary (assuming it's categorical)
    if dataframe['is_safe'].dtype == 'object' or dataframe['is_safe'].isnull().any():
        label_encoder = LabelEncoder()
        dataframe['is_safe'] = label_encoder.fit_transform(dataframe['is_safe'].fillna(0))

    return dataframe

# Function for user input in Heart Attack model
def get_heart_attack_input():
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, value=100)
    diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction", min_value=0, max_value=100, value=50)
    high_blood_pressure = st.selectbox("High Blood Pressure (0: No, 1: Yes)", [0, 1])
    platelets = st.number_input("Platelets", min_value=0.0, value=250000.0)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.0)
    serum_sodium = st.number_input("Serum Sodium", min_value=0, value=140)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])
    time = st.number_input("Follow-up Period (days)", min_value=0, value=100)
    
    features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium, 
                          sex, smoking, time]])
    return features

# Function for user input in Water Quality model
def get_water_quality_input():
    aluminium = st.number_input("Aluminium", min_value=0.0, value=0.0)
    ammonia = st.number_input("Ammonia", min_value=0.0, value=0.0)
    arsenic = st.number_input("Arsenic", min_value=0.0, value=0.0)
    barium = st.number_input("Barium", min_value=0.0, value=0.0)
    cadmium = st.number_input("Cadmium", min_value=0.0, value=0.0)
    chloramine = st.number_input("Chloramine", min_value=0.0, value=0.0)
    chromium = st.number_input("Chromium", min_value=0.0, value=0.0)
    copper = st.number_input("Copper", min_value=0.0, value=0.0)
    fluoride = st.number_input("Fluoride", min_value=0.0, value=0.0)
    bacteria = st.number_input("Bacteria", min_value=0.0, value=0.0)
    viruses = st.number_input("Viruses", min_value=0.0, value=0.0)
    lead = st.number_input("Lead", min_value=0.0, value=0.0)
    nitrates = st.number_input("Nitrates", min_value=0.0, value=0.0)
    nitrites = st.number_input("Nitrites", min_value=0.0, value=0.0)
    mercury = st.number_input("Mercury", min_value=0.0, value=0.0)
    perchlorate = st.number_input("Perchlorate", min_value=0.0, value=0.0)
    radium = st.number_input("Radium", min_value=0.0, value=0.0)
    selenium = st.number_input("Selenium", min_value=0.0, value=0.0)
    silver = st.number_input("Silver", min_value=0.0, value=0.0)
    uranium = st.number_input("Uranium", min_value=0.0, value=0.0)

    features = np.array([[aluminium, ammonia, arsenic, barium, cadmium, chloramine,
                          chromium, copper, fluoride, bacteria, viruses, lead,
                          nitrates, nitrites, mercury, perchlorate, radium,
                          selenium, silver, uranium]])
    return features

# Main app
def main():
    st.title("Model Selection and Evaluation")

    # Add a sidebar for model selection
    model_choice = st.sidebar.selectbox("Choose the Model", ("Heart Failure Survival Model", "Water Quality Safety Model"))

    if model_choice == "Heart Failure Survival Model":
        # Upload file
        uploaded_file = st.file_uploader("Upload your Heart Failure CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load the dataset
            st.write("Loading the dataset...")
            dataframe = load_heart_data(uploaded_file)

            # Display the first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Tabs for Heart Attack model evaluation
            tabs = st.tabs(["K-fold Cross Validation", "Leave-One-Out Cross Validation", "Prediction"])

            with tabs[0]:
                st.subheader("K-fold Cross Validation")
                array = dataframe.values
                X = array[:, :-1]  # Features
                Y = array[:, -1]   # Target variable
                num_folds = st.slider("Select number of folds for KFold Cross Validation:", 2, 10, 5)
                kfold = KFold(n_splits=num_folds)
                model = LogisticRegression(max_iter=210)
                results = cross_val_score(model, X, Y, cv=kfold)
                st.write(f"Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                # Metrics Calculation
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
                model.fit(X_train, Y_train)
                Y_prob = model.predict_proba(X_test)[:, 1]
                
                # Classification Metrics
                st.subheader("Classification Metrics")
                st.write("Confusion Matrix:")
                predicted = model.predict(X_test)
                matrix = confusion_matrix(Y_test, predicted)
                st.write(matrix)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=matrix).plot(cmap=plt.cm.Blues, ax=ax)
                st.pyplot(fig)

                st.write("Classification Report:")
                report = classification_report(Y_test, predicted, output_dict=True)
                st.write(report)

                st.write(f"ROC AUC Score: {roc_auc_score(Y_test, Y_prob):.3f}")

                # Plot ROC Curve
                fpr, tpr, _ = roc_curve(Y_test, Y_prob)
                plt.figure()
                plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_score(Y_test, Y_prob))
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                st.pyplot()

                st.write(f"Logarithmic Loss: {log_loss(Y_test, Y_prob):.3f}")

                # Classification Accuracy
                accuracy = accuracy_score(Y_test, predicted)
                st.write(f"Classification Accuracy: {accuracy * 100:.3f}%")

                # Save the trained model
                model_filename = "heart_attack_kfold_model.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename)

            with tabs[1]:
                st.subheader("Leave-One-Out Cross Validation (LOOCV)")
                loocv = LeaveOneOut()
                model = LogisticRegression(max_iter=500)
                results = cross_val_score(model, X, Y, cv=loocv)
                st.write(f"Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                # Metrics Calculation
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
                model.fit(X_train, Y_train)
                Y_prob = model.predict_proba(X_test)[:, 1]
                
                # Classification Metrics
                st.subheader("Classification Metrics")
                st.write("Confusion Matrix:")
                predicted = model.predict(X_test)
                matrix = confusion_matrix(Y_test, predicted)
                st.write(matrix)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=matrix).plot(cmap=plt.cm.Blues, ax=ax)
                st.pyplot(fig)

                st.write("Classification Report:")
                report = classification_report(Y_test, predicted, output_dict=True)
                st.write(report)

                st.write(f"ROC AUC Score: {roc_auc_score(Y_test, Y_prob):.3f}")

                # Plot ROC Curve
                fpr, tpr, _ = roc_curve(Y_test, Y_prob)
                plt.figure()
                plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_score(Y_test, Y_prob))
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                st.pyplot()

                st.write(f"Logarithmic Loss: {log_loss(Y_test, Y_prob):.3f}")

                # Classification Accuracy
                accuracy = accuracy_score(Y_test, predicted)
                st.write(f"Classification Accuracy: {accuracy * 100:.3f}%")

                # Save the trained model
                model_filename = "heart_attack_loocv_model.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename)

            with tabs[2]:
                st.subheader("Prediction")
                st.write("Upload your trained model for prediction:")
                uploaded_model = st.file_uploader("Upload your trained model file", type=["pkl"])
                user_data = get_heart_attack_input()
                if uploaded_model is not None:
                    loaded_model = joblib.load(uploaded_model)
                    prediction = loaded_model.predict(user_data)
                    st.write("Prediction Result:")
                    st.write("Survival" if prediction[0] == 0 else "Death")

    elif model_choice == "Water Quality Safety Model":
        # Upload file
        uploaded_file = st.file_uploader("Upload your Water Quality CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load the dataset
            st.write("Loading the dataset...")
            dataframe = load_water_data(uploaded_file)

            # Display the first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Tabs for Water Quality Safety model evaluation
            tabs = st.tabs(["Train-Test Split", "Repeated Random Test-Train Splits", "Prediction"])

            with tabs[0]:
                st.subheader("Split into Train and Test Sets")
                test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
                X = dataframe.drop('is_safe', axis=1).values
                Y = dataframe['is_safe'].values
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
                model = LogisticRegression(max_iter=200)
                model.fit(X_train, Y_train)
                result = model.score(X_test, Y_test)
                st.write(f"Accuracy: {result * 100:.3f}%")

                # Regression Metrics
                st.subheader("Regression Metrics")
                Y_pred = model.predict(X_test)
                mse = mean_squared_error(Y_test, Y_pred)
                st.write(f"MSE: {mse:.3f}")

                mae = np.mean(np.abs(Y_test - Y_pred))
                st.write(f"MAE: {mae:.3f}")

                r2 = 1 - (np.sum((Y_test - Y_pred) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2))
                st.write(f"R-squared: {r2:.3f}")

                # Save the trained model
                model_filename = "water_quality_splitmodel.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename)

            with tabs[1]:
                st.subheader("Repeated Random Test-Train Splits")
                n_splits = st.slider("Select number of splits:", 2, 20, 10)
                test_size = st.slider("Select test size proportion:", 0.1, 0.5, 0.33)
                shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size)

                # Fit the model and evaluate
                model = LogisticRegression(max_iter=300)
                results = cross_val_score(model, X, Y, cv=shuffle_split)
                st.write(f"Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                # Fit the model on the complete dataset for regression metrics
                model.fit(X, Y)
                Y_pred = model.predict(X_test)

                # Regression Metrics
                st.subheader("Regression Metrics")
                mse = mean_squared_error(Y_test, Y_pred)
                st.write(f"MSE: {mse:.3f}")

                mae = np.mean(np.abs(Y_test - Y_pred))
                st.write(f"MAE: {mae:.3f}")

                r2 = 1 - (np.sum((Y_test - Y_pred) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2))
                st.write(f"R-squared: {r2:.3f}")

                # Save the trained model
                model_filename = "water_quality_repeatedmodel.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename)

            with tabs[2]:
                st.subheader("Prediction")
                st.write("Upload your trained model for prediction:")
                uploaded_model = st.file_uploader("Upload your trained model file", type=["pkl"])
                user_data = get_water_quality_input()
                if uploaded_model is not None:
                    loaded_model = joblib.load(uploaded_model)
                    prediction = loaded_model.predict(user_data)
                    st.write("Prediction Result:")
                    st.write("Safe" if prediction[0] == 0 else "Not Safe")

if __name__ == "__main__":
    main()
