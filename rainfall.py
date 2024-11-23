import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

import warnings
warnings.filterwarnings('ignore')
print("importin")
data=pd.read_csv('rainfall in india 1901-2015.csv')
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Fill missing values with the median of the respective columns for numeric columns only
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
features = data.columns.difference(['SUBDIVISION', 'YEAR', 'ANNUAL'])
X = data[features]
y = data['ANNUAL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')
joblib_file = "rainfall_model.pkl"
joblib.dump(model, joblib_file)
print(f"Model saved to {joblib_file}")
#print(df.head())
