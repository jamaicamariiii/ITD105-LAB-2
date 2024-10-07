Features
1. Heart Failure Survival Prediction Model
Dataset Input:
        Users can upload a CSV file containing patient information related to heart failure.
Model Evaluation:
        Users can perform K-Fold Cross Validation and Leave-One-Out Cross Validation (LOOCV).
Metrics:
        Confusion Matrix
        Classification Report
        ROC AUC Curve
        Logarithmic Loss
        Accuracy Score
Model Download:
        The trained Logistic Regression model can be downloaded for further use.
Prediction:
        Users can upload a trained model and input new data to get survival predictions.

   
2. Water Quality Safety Prediction Model
Dataset Input:
        Users can upload a CSV file containing water quality data.
Model Evaluation:
        Available methods include Train-Test Split and Repeated Random Test-Train Splits.
Metrics:
        Mean Squared Error (MSE)
        Mean Absolute Error (MAE)
        R-squared (R²)
Prediction:
        Users can upload a trained model to predict the safety of water samples.
   
3. Model Persistence
Both the Heart Failure Survival and Water Quality Safety models allow the trained models to be saved as .pkl files using joblib. Users can then upload and reuse these models for prediction purposes.
      
