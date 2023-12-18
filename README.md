# Cleared volume forecasting

This project evaluates and selects the best forecasting model for predicting volume based on historical data. The models considered include baseline models, rolling averages, Prophet, Auto ARIMA, and TFT.

## Getting Started

These instructions will guide you through setting up and running the project on your local machine.

### Prerequisites

Make sure you have the following installed:

- Python (version 3.9^)
- [Pip](https://pip.pypa.io/en/stable/installation/)

### Installing Dependencies

Install the required Python libraries using the following command:

```bash
poetry install
```

### Running the Code

Execute the main code by running the following command:

```bash
poetry shell
python volume_prediction/main.py
```

## Code Structure

The project code is organized into modules for better readability and maintainability.

- **help_functions:** Contains utility functions for handling outliers, reading configurations, and loading data.
- **models:** Includes different forecasting models such as baseline, rolling average, Prophet, Auto ARIMA, and TFT.
- **evaluation:** Provides functions for calculating Mean Absolute Percentage Error (MAPE).
- **tft_model:** Defines the TFT model for forecasting.

## Configuration

The `config.json` file contains configuration parameters such as input/output file paths, model settings, and date ranges. Adjust these values as needed for your specific use case.

## Results

The project evaluates each model's performance using MAPE and outputs the results to a CSV file. The best-performing model is selected based on the lowest MAPE.

## Authors

- Roberto Félix Patrón

