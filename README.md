# CNN-LSTM for Inflation Forecasting

## Overview

This project explores the use of a Convolutional Neural Network combined with Long Short-Term Memory (CNN-LSTM) networks for forecasting inflation, specifically using the Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL) as the target variable.  The project aims to demonstrate the feasibility of deep learning models for time series forecasting in economics, comparing its performance against traditional statistical methods like VAR, ARIMA, and SARIMA.

## Models Implemented

This repository contains implementations and evaluations of the following time series forecasting models:

1.  **Vector Autoregression (VAR):** A classical multivariate time series model used as a benchmark.
2.  **Autoregressive Integrated Moving Average (ARIMA):** A univariate time series model adapted for a single variable forecast in this context.
3.  **Seasonal Autoregressive Integrated Moving Average (SARIMA):** An extension of ARIMA to handle seasonality in time series data.
4.  **Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM):** A deep learning model that combines convolutional layers for feature extraction and LSTM layers for sequence learning. This is the primary focus of the project.

## Data

The project uses the publicly available **CPIAUCSL** dataset, sourced from FRED (Federal Reserve Economic Data).  Optionally, other economic indicators could be included as exogenous variables to enhance forecasting accuracy, especially for multivariate models like VAR and CNN-LSTM.

**Data Preprocessing Steps:**

*   **Data Loading:** Reading the time series data from a CSV file (e.g., `data.csv`).
*   **Scaling:** Applying MinMaxScaler to scale the numerical features and target variable to the range [0, 1]. This is crucial for training neural networks effectively.
*   **Train-Test Split:** Dividing the data into training and testing sets. A fixed test size (e.g., 12 periods) is used to evaluate the models' forecasting performance on recent data.  The training set is used to fit the models, and the test set is used to evaluate out-of-sample forecast accuracy.
*   **Reshaping for CNN-LSTM:**  Reshaping the input data into a 5D tensor format `(samples, time_steps, rows, cols, features)` suitable for ConvLSTM2D layers. In this project, `time_steps`, `rows`, and `cols` are set to 1 for simplicity, focusing on feature dimension.

## CNN-LSTM Model Architecture

The base CNN-LSTM model implemented in this project has the following architecture:

*   **Bidirectional ConvLSTM2D Layers:**
    *   First Layer: 128 filters, kernel size (1, 1), 'selu' activation, `return_sequences=True`, input shape `(1, 1, 1, features)` (where `features` is the number of input features).
    *   Second Layer: 64 filters, kernel size (1, 1), 'selu' activation, `return_sequences=False`.
    *   Bidirectional wrappers are used to process the sequence in both forward and backward directions, potentially capturing more complex temporal patterns.
    *   'selu' activation function is used for potential self-normalizing properties.
*   **Flatten Layer:** Flattens the output from the ConvLSTM2D layers into a 1D tensor.
*   **Dense Layer:** Output layer with a single neuron (for univariate forecasting) and linear activation (default).
*   **Optimizer:** Adam optimizer with a learning rate of 0.0005.
*   **Loss Function:** Mean Squared Error (MSE) is used as the loss function, suitable for regression tasks like time series forecasting.
*   **Early Stopping:**  Implemented using `tensorflow.keras.callbacks.EarlyStopping` to monitor validation loss (`val_loss`) and stop training if validation loss does not improve for a patience of 50 epochs, restoring the best weights to prevent overfitting.

This architecture will change as we experiment with other hyperparameters.

## Code Structure and Key Functions

The project code is structured into several Python functions for modularity and readability:

*   **`prepare_data(data, target_column, test_size, test_size_percent)`:**  Handles data splitting, scaling, and reshaping for the CNN-LSTM model. Returns `X_train`, `y_train`, `X_test`, `y_test`.
*   **`build_model(input_shape)`:**  Defines and compiles the CNN-LSTM model architecture. Returns the compiled Keras model.
*   **`train_model(model, X_train, y_train)`:** Trains the provided model using `X_train`, `y_train` with early stopping. Returns the training `history` object.
*   **`evaluate_model(model, X_test, y_test)`:** Evaluates the trained model on the test set and returns the test loss and predictions.
*   **`inverse_transform(scaler, data)`:**  Inverse transforms scaled data back to the original scale using the provided scaler.
*   **`plot_predictions(y_actual, y_pred, title)`:** Generates a plot comparing actual vs forecasted values on the test set.
*   **`plot_observed_vs_fitted(model, train_data_y, model_type, fitted_values=None, target_column, title)`:**  Plots observed vs fitted values on the training set for different model types (VAR, ARIMA, CNN-LSTM).
*   **`plot_loss(history)`:** Plots the training and validation loss curves from the training history.
*   **`plot_forecast(forecasted_series, actual_series, title_name)`:** A generalized plotting function for forecast vs actual series with enhanced visual appeal and date formatting.
*   **`plot_grid_series(figures, titles, grid_rows, grid_cols, figsize)`:** Plots a grid of matplotlib figures, allowing for comparison of results from different models in a single figure.
*   **`main(data, scaler_y)`:** The main function that orchestrates the entire pipeline: data preparation, model building, training, evaluation, prediction, and plotting.

## Results and Evaluation

The models are evaluated primarily using:

*   **Test Loss (MSE):** Mean Squared Error on the test dataset.
*   **Mean Absolute Error (MAE):**  Average absolute difference between actual and forecasted values on the test set.
*   **Root Mean Squared Error (RMSE):** Square root of the MSE on the test set, providing an error metric in the original units of the target variable.

**Visualizations:**

*   **Actual vs Forecasted Values Plot:** Shows the forecasted CPIAUCSL against the actual CPIAUCSL values for the test period.
*   **Observed vs Fitted Values Plot:**  Displays the model's in-sample "fitted" values compared to the actual CPIAUCSL values during the training period. This plot is generated for each model (VAR, ARIMA, CNN-LSTM).
*   **Training Loss Plot:**  Illustrates the training and validation loss curves over epochs for the CNN-LSTM model, helping to assess model convergence and potential overfitting.
*   **Grid Plot:** A combined figure presenting all key plots (forecasts and fitted values from different models) in a 2x4 grid for easy visual comparison.

**Expected Outcomes:**

The CNN-LSTM model is expected to capture complex temporal patterns in the inflation data and potentially outperform traditional linear models like VAR, ARIMA, and SARIMA, especially in capturing non-linear dynamics.  The project will compare the quantitative metrics (MAE, RMSE) and visual outputs to assess the relative performance of these models.

## How to Run the Code

1.  **Prerequisites:**
    *   **Python 3.x**
    *   **Libraries:** Install required Python libraries using pip:
        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
        ```
        or using conda:
        ```bash
        conda install pandas numpy scikit-learn matplotlib seaborn tensorflow
        ```

2.  **Data File:** Ensure you have the CPIAUCSL dataset (e.g., in a `data.csv` file in the same directory as the script) or modify the data loading part of the script to point to your data source.

3.  **Run the Main Script:** Execute the main Python script (e.g., `main_script.py`) from your terminal:
    ```bash
    python main_script.py
    ```

4.  **Output:** The script will:
    *   Train the CNN-LSTM model and potentially other models (if implemented in `main_script.py`).
    *   Evaluate the models on the test set and print evaluation metrics (Test Loss, MAE, RMSE).
    *   Generate and display various plots: forecast plots, observed vs fitted plots, and loss curves.
    *   If implemented, it will also generate a grid plot combining visualizations from different models.

## Possible Improvements and Future Work

*   **Hyperparameter Tuning:**  Perform more extensive hyperparameter tuning for the CNN-LSTM model (e.g., number of layers, filters, kernel sizes, learning rate, batch size, epochs, optimizer choices) to potentially improve performance.
*   **Feature Engineering:** Incorporate additional relevant economic indicators (e.g., unemployment rate, interest rates, GDP growth, etc.) as exogenous variables to potentially improve forecast accuracy, especially for the VAR and CNN-LSTM models (making them multivariate).
*   **Advanced CNN-LSTM Architectures:** Experiment with different CNN-LSTM architectures, such as adding attention mechanisms, more complex convolutional or LSTM layer configurations, or exploring different types of recurrent layers (GRU).
*   **Model Ensembling:** Explore ensemble methods, combining predictions from multiple models (e.g., averaging forecasts from CNN-LSTM, VAR, ARIMA) to potentially create more robust and accurate forecasts.
*   **Evaluation Metrics:** Explore and report additional evaluation metrics beyond MAE and RMSE, such as MAPE (Mean Absolute Percentage Error), sMAPE (Symmetric Mean Absolute Percentage Error), or metrics relevant to economic forecasting.
*   **Real-time Forecasting and Deployment:**  Consider deploying the best performing model for real-time inflation forecasting, potentially setting up an automated pipeline for data ingestion, model retraining, and forecast generation.
