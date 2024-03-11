# %%
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib


# %%
# Define column names
columns = ['year', 'month', 'day', 'hour', 'temperature', 'precipitation', 'u-wind', 'v-wind']

# Load data from CSV without headers
df = pd.read_csv("17.36N_78.5E.csv", names=columns)
df['temperature'] -= 273.15

# Take a subset of the data for faster testing
percentage = 0.4
num_rows = len(df)
top_rows = int(num_rows * percentage)
df = df.head(top_rows)


# %%
# Feature engineering
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df['season'] = df['datetime'].dt.month // 3 + 1
df['time_of_day'] = pd.cut(df['hour'], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3])
df['day_of_week'] = df['datetime'].dt.day_of_week
df['time_since_start'] = (df['datetime'] - df['datetime'].min()).dt.days
df.drop(columns=['datetime'], inplace=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['temperature']),
    df['temperature'],
    test_size=0.2,
    random_state=1
)



# %%
# Define parameter grid for grid search
param_grid = {
    'hidden_layer_sizes': [(128,), (64, 32), (64, 64, 32), (128, 64, 32, 16)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'adam'],
    'max_iter': [100, 200, 250, 300]
}

# Initialize MLPRegressor
mlp = MLPRegressor()

# Perform grid search
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)



# %%
# Get the best parameters
best_params = grid_search.best_params_

# Train the best model using the best parameters
best_model = MLPRegressor(**best_params)
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
test_loss = mean_squared_error(y_test, y_pred)

# Report test loss
print("Test Loss:", test_loss)
print("Best Parameters:", best_params)

# Save the best model
joblib.dump(best_model, 'ADS_Grid_Model.pkl')



