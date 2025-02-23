# Air_Quality_Prediction
# Indian Air Quality Analysis & Prediction using ML

## Introduction
This project aims to analyze and predict air quality in India using machine learning techniques. It utilizes historical air quality data to identify trends, assess pollution levels, and forecast future air quality index (AQI) values.

## Features
- Data preprocessing and handling missing values
- Exploratory data analysis (EDA) for insights
- Feature engineering and selection
- Machine learning models for AQI prediction
- Performance evaluation and visualization

## Dataset
The dataset used for this project is a historical air quality dataset, loaded as follows:

```python
import pandas as pd

data = pd.read_csv('../input/india-air-quality-data/data.csv', encoding="ISO-8859-1")
data.fillna(0, inplace=True)
data.head()
```

### Dataset Attributes
The dataset includes various attributes such as:
- **Date & Time**: Timestamp of the recorded data
- **PM2.5, PM10**: Particulate matter concentrations
- **NO2, SO2, CO, O3**: Concentrations of gaseous pollutants
- **AQI**: Air Quality Index value

## Installation
To set up the project environment, install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage
1. **Data Preprocessing**: Handle missing values, convert date-time formats, and clean anomalies.
2. **Exploratory Data Analysis**: Visualize pollutant trends and correlations.
3. **Feature Engineering**: Select relevant features for prediction.
4. **Train Machine Learning Models**: Implement regression models like Random Forest, Linear Regression, and LSTMs.
5. **Evaluate Performance**: Use metrics like RMSE, MAE, and R²-score.
6. **Make Predictions**: Predict future AQI values based on trained models.

## Example Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
y = data['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## Future Improvements
- Implement deep learning models for better accuracy.
- Integrate real-time AQI prediction APIs.
- Deploy the model as a web application.

## License
This project is open-source and available under the MIT License.

## Contributors
   P.Karunakar

## Acknowledgments
- Data sourced from Indian air quality monitoring agencies.
- Inspired by research in environmental data science.

