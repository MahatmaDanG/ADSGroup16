# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--mlp", action="store_true")
parser.add_argument("--rf", action="store_true")
args = parser.parse_args()

# %%
data_dir = '../metdata'

files = os.listdir(data_dir)
files = [os.path.join(data_dir,x) for x in files]
files

pattern = r"([-]?\d+[\.]?\d+)N_([-]?\d+[\.]?\d+)E\.csv"
parsed_latlon = []
for file in files:
    match = re.search(pattern, file)
    if match:
        latitude = float(match.group(1))
        longitude = float(match.group(2))
        if longitude > 180: longitude -= 360 
    parsed_latlon.append((float(latitude), float(longitude)))
lat, lon = zip(*parsed_latlon)

locations_dict = {
    "40.75": "New York",
    "51.5": "London",
    "59.92": "Oslo",
    "43.28": "Marseille",
    "17.36": "Hyderabad",
    "41.9": "Rome",
    "58.76": "Churchill",
    "51.03": "Calgary",
    "-33.5": "Sydney",
    "43.64": "Toronto",
    "-33.9": "Cape Town",
    "18.0": "Kingston"
}

kelvin_to_cels = -273.15


# %%

# %%
def readfile(path, index):
    df = pd.read_csv(path)
    headings = ["year", "month", "day", "hour", "temperature", "precipitation", "u-wind", "v-wind"]
    df.columns = headings
    df["temperature"] = df["temperature"] + kelvin_to_cels
    # df["wind-speed"] = np.sqrt(df["u-wind"]**2 + df["v-wind"]**2)
    df["longitude"] = lon[index]
    df["latitude"] = lat[index]
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df



# %%
def join_datasets(files):
    dfs = []
    for i, file in enumerate(files):
        df = readfile(file, i)
        dfs.append(df)
    dfs = pd.concat(dfs, axis = 0)
    dfs.reset_index(drop=True, inplace=True)
    return dfs

dfs = join_datasets(files)
# %%

idx = 0
df_index = readfile(files[idx], idx)
location = locations_dict[str(lat[idx])]

# %%
df_index.describe().transpose()

# %%
import seaborn as sns

target_columns = ["temperature", "precipitation", "u-wind", "v-wind"]

units_dict = {
    'temperature': '°C',        
    'precipitation': 'mm', 
    'u-wind': 'm/s',  
    'v-wind': 'm/s'              
}

# %%
df_size = len(df_index)
train_size = int(df_size * 0.8)
val_end = int(df_size * 0.15) + train_size
df_train = df_index[:train_size]
df_val = df_index[train_size:val_end]
df_test = df_index[val_end:]

# %%
target = "temperature"

df_avg = df_train.groupby(['month','day', 'hour'])[target].mean().reset_index()
df_avg.columns = ['month', 'day', 'hour', f'avg_{target}']
df_avg_full =   pd.merge(df_index, df_avg, on=['month', 'day', 'hour'], how='left')
df_avg_test = pd.merge(df_test, df_avg, on=['month','day', 'hour'], how='left')
# %%

df_avg_full.shape
# %%
timespan_to_plot = 2000
plt.clf()
plt.figure(figsize=(10,6))
plt.plot(df_test['datetime'][:timespan_to_plot], df_test[target][:timespan_to_plot], label='Actual Temperature', color='blue', linewidth=1)
plt.plot(df_test['datetime'][:timespan_to_plot], df_avg_test[f"avg_{target}"][:timespan_to_plot], label='Estimated Temperature', color='red', linestyle='--', linewidth=1)

plt.title(f'Actual vs Estimated {target.capitalize()} in {location}')
plt.xlabel('Date and Time')
plt.ylabel(f'{target.capitalize()} ({units_dict[target]})')

# Add legend
plt.legend()

# %% 

from sklearn.metrics import mean_absolute_error, mean_squared_error

actual = df_test[target]
predicted = df_avg_test[f'avg_{target}']

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

print("Averaging Results:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# %%

df_rolling = df_index.copy()
group_id = (df_rolling.index // 6)
df_rolling[f'rolling_{target}'] = group_id.map(df_rolling.groupby(group_id)[target].first())
df_rolling
# %%
timespan_to_plot = 2000
plt.clf()
plt.figure(figsize=(10,6))
plt.plot(df_test['datetime'][:timespan_to_plot], df_test['temperature'][:timespan_to_plot], label='Actual Temperature', color='blue', linewidth=1)
plt.plot(df_test['datetime'][:timespan_to_plot], df_rolling[f"rolling_{target}"][val_end:][:timespan_to_plot], label='Estimated Temperature', color='red', linestyle='--', linewidth=1)

plt.title(f'Actual vs Estimated {target.capitalize()} in {location}')
plt.xlabel('Date and Time')
plt.ylabel(f'{target.capitalize()} ({units_dict[target]})')

plt.legend()

# %%

actual = df_test[target]
predicted = df_rolling[f'rolling_{target}'][val_end:]

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

print("Rolling Results:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# %%

df_persistence = df_index.copy()
df_persistence[f'persistence_{target}'] = df_persistence['temperature'].shift(24)
df_persistence
# df_persistence = df_persistence.dropna(subset=[f"persistence_{target}"])




# %%
timespan_to_plot = 200
plt.clf()
plt.figure(figsize=(10,6))
plt.plot(df_test['datetime'][:timespan_to_plot], df_test['temperature'][:timespan_to_plot], label='Actual Temperature', color='blue', linewidth=1)
plt.plot(df_test['datetime'][:timespan_to_plot], df_persistence[f"persistence_{target}"][val_end:][:timespan_to_plot], label='Estimated Temperature', color='red', linestyle='--', linewidth=1)

plt.title(f'Actual vs Estimated {target.capitalize()} in {location}')
plt.xlabel('Date and Time')
plt.ylabel(f'{target.capitalize()} ({units_dict[target]})')

plt.legend()
# %%
actual = df_test[target]
predicted = df_persistence[f'persistence_{target}'][val_end:]

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

print("Persistence Results:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
# %%

timespan_to_plot = 2000
plt.clf()
plt.figure(figsize=(10,6))
plt.plot(df_test['datetime'][:timespan_to_plot], df_test['temperature'][:timespan_to_plot], label='Actual Temperature', color='blue', linewidth=2)
plt.plot(df_test['datetime'][:timespan_to_plot], df_persistence[f"persistence_{target}"][val_end:][:timespan_to_plot], label='Persistence Model Prediction', color='orange', linestyle='--', linewidth=2)

plt.plot(df_test['datetime'][:timespan_to_plot], df_rolling[f"rolling_{target}"][val_end:][:timespan_to_plot], label='Rolling Model Prediction', color='green', linestyle='--', linewidth=2)

plt.plot(df_test['datetime'][:timespan_to_plot], df_avg_test[f"avg_{target}"][:timespan_to_plot], label='Averaging Model Prediction', color='red', linestyle='--', linewidth=2)
plt.title(f'Actual vs Estimated {target.capitalize()} in {location}')
plt.xlabel('Date and Time')
plt.ylabel(f'{target.capitalize()} ({units_dict[target]})')

plt.legend()

# %%

#NOW ONTO PROPER ML METHODS

df_mlp = df_index.copy()

datetime = df_mlp["datetime"]

timestamp_s = df_mlp["datetime"].map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day

df_mlp['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df_mlp['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df_mlp['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df_mlp['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

features_to_drop = ["datetime", "latitude", "longitude",
                    "year", "month", "day", "hour"]
for to_drop in features_to_drop:
    df_mlp = df_mlp.drop(to_drop, axis = 1)
target_shift = -6 
df_mlp[f'shift_{target}'] = df_mlp['temperature'].shift(target_shift)
df_mlp = df_mlp.dropna(subset=f'shift_{target}')
features = df_mlp.columns

X_features = features.drop(f'shift_{target}')
y_feature = f'shift_{target}'



X = df_mlp[X_features]
y = df_mlp[y_feature]
X = X.to_numpy()
y = y.to_numpy()
# %%

X_train, X_val, X_test = X[:train_size], X[train_size:val_end], X[val_end:]

y_train, y_val, y_test = y[:train_size], y[train_size:val_end], y[val_end:]

# %%

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.fit_transform(X_test)
X_val_scaled = scaler.fit_transform(X_val)

# %%

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

# mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', 
#                    max_iter=500, random_state=8, verbose = True, early_stopping=True)
# mlp = MLPRegressor(max_iter=500, early_stopping=True)
# param_grid_mlp = {
#     'hidden_layer_sizes': [(25,25,25),(50,), (100,), (50, 50)],  # Different configurations of layers
#     'activation': ['tanh', 'relu'],                   # Activation function for the hidden layer
#     'solver': ['sgd', 'adam'],                        # The solver for weight optimization
#     'alpha': [0.0001, 0.05],                          # Regularization term
#     'learning_rate': ['constant','adaptive']          # Learning rate schedule for weight updates
# }

# # Train the model
# # mlp.fit(X_train_scaled, y_train)
# grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
# grid_search_mlp.fit(X_train_scaled, y_train)

# print("Best parameters:", grid_search_mlp.best_params_)
# print("Best RMSE:", np.sqrt(-grid_search_mlp.best_score_))
#%%
mlp = MLPRegressor(hidden_layer_sizes=(50,50), activation="relu", solver="sgd",
                   alpha=0.05, learning_rate="adaptive", verbose = True, max_iter=500)
mlp.fit(X_train_scaled, y_train)
# %%
y_pred_mlp = mlp.predict(X_test_scaled)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred_mlp)
mae = mean_absolute_error(y_test, y_pred_mlp)

print("MLPRegressor Scores:")
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

plt.clf()
plt.figure(figsize=(8, 4))
plt.plot(mlp.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
# %%

timespan_to_plot = 200
plt.clf()
plt.figure(figsize=(12, 6))

new_test_start = val_end - target_shift
end_window = new_test_start + timespan_to_plot
# Plot actual temperatures
plt.plot(datetime[new_test_start:end_window], y_test[:timespan_to_plot], label='Actual Temperature', color='blue', linestyle='-', linewidth=1)

# Plot predicted temperatures
plt.plot(datetime[new_test_start:end_window], y_pred_mlp[:timespan_to_plot], label='Predicted Temperature', color='red', linestyle='--', linewidth=1)

plt.title(f'Actual vs Estimated {target.capitalize()} in {location}')
plt.xlabel('Date and Time')
plt.ylabel(f'{target.capitalize()} ({units_dict[target]})')
plt.legend()

plt.show()


# %%

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid_rf = {
    'n_estimators': [50, 100],  # Number of trees in the forest
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'max_depth': [10, 20, 30, None],   # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required at each leaf node
}

rf = RandomForestRegressor(random_state=8, verbose=1) 
# rf.fit(X_train_scaled, y_train)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=None, verbose=2, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search_rf.best_params_)
print("Best RMSE:", np.sqrt(-grid_search_rf.best_score_))

# %%
y_pred_rf = rf.predict(X_test_scaled)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred_rf)
mae = mean_absolute_error(y_test,y_pred_rf)

print("RandomForestRegressor Scores:")
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
# %%
import hmmlearn


# %%

timespan_to_plot = 200
plt.clf()
plt.figure(figsize=(12, 6))

new_test_start = val_end - target_shift
end_window = new_test_start + timespan_to_plot
# Plot actual temperatures
plt.plot(datetime[new_test_start:end_window], y_test[:timespan_to_plot], label='Actual Temperature', color='blue', linestyle='-', linewidth=1)

# Plot predicted temperatures
# plt.plot(datetime[new_test_start:end_window], y_pred_mlp[:timespan_to_plot], label='MLP Predicted Temperature', color='green', linestyle='--', linewidth=1)

plt.plot(datetime[new_test_start:end_window], y_pred_rf[:timespan_to_plot], label='RF Predicted Temperature', color='orange', linestyle='--', linewidth=1)

plt.title(f'Actual vs Estimated {target.capitalize()} in {location}')
plt.xlabel('Date and Time')
plt.ylabel(f'{target.capitalize()} ({units_dict[target]})')
plt.legend()

plt.show()
# %%

import tensorflow as tf
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

# train_df = df.

# %%
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df=train_df, val_df=val_df, test_df=test_df,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        print(self.input_indices, self.label_indices)

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model=None, plot_col='temperature', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        # print("labels shape: ", labels.shape, "\nlabels: ", labels)
        for n in range(max_n):
            # print("!!!\n n \n !!!")
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue
            # print("length of label _indices ", len(self.label_indices))
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                # print("predictions shape: ", predictions.shape, "\npredictions: ", predictions[n, :, label_col_index].shape)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time (hours)')
        plt.savefig(f'window_{plot_col}_{self.input_width}_{self.label_width}_{self.shift}.png')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds
    
    def train(self):
        return self.make_dataset(self.train_df)

    def val(self):
        return self.make_dataset(self.val_df)

    def test(self):
        return self.make_dataset(self.test_df)

    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
  
target ="temperature"
w1 = WindowGenerator(input_width=24, label_width=24, shift=6,
                     label_columns=[target])

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                           np.array(train_df[100:100+w1.total_window_size]),
                           np.array(train_df[200:200+w1.total_window_size])])

example_inputs, example_labels = w1.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

w1.example = example_inputs, example_labels

# w1.plot()
# dataset.get_batch([0,1,2,3])

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


# %%

print('Input shape:', w1.example[0].shape)
print('Output shape:', lstm_model(w1.example[0]).shape)

MAX_EPOCHS = 20

