# %% [markdown]
# # Attempt based on Google's GraphCast
# 
# Their model takes input of weather, and predicts the weather 6 hours later

# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

data_dir = '../metdata'

files = os.listdir(data_dir)
files = [os.path.join(data_dir,x) for x in files]
files

pattern = r"([-]?\d+[\.]?\d+)N_([-]?\d+[\.]?\d+)E\.csv"
parsed_latlon = []
for file in files:
    match = re.search(pattern, file)
    if match:
        latitude = match.group(1)
        longitude = match.group(2)
    parsed_latlon.append((float(latitude), float(longitude)))
lat, lon = zip(*parsed_latlon)

kelvin_to_cels = -273.15



# %%
def readfile(path, index):
    df = pd.read_csv(path)
    headings = ["year", "month", "day", "hour", "temperature(celsius)", "precipitation", "u-wind", "v-wind"]
    df.columns = headings
    df["temperature(celsius)"] = df["temperature(celsius)"] + kelvin_to_cels
    df["wind-speed"] = np.sqrt(df["u-wind"]**2 + df["v-wind"]**2)
    df["longitude"] = lon[index]
    df["latitude"] = lat[index]
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    # df['week_num'] = df['datetime'].dt.isocalendar().week
    df['week_num'] = np.floor((df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / (60 * 60 * 24 * 7))
    # df['fortnight'] = np.floor(df['week_num'] / 2)
    # df['day_num'] = np.floor((df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / (60 * 60 * 24))
    df['day_num'] = df['datetime'].dt.dayofyear
    # df['X'] = df['week_num'] % 2 == 0
    df['hour_sin'] = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2. * np.pi / 24))
    min_year = np.min(df['year'])
    max_year = np.max(df['year'])
    df['year'] = (df["year"] - min_year) / (max_year - min_year)
    df['day_of_year_sin'] = np.sin(df['day_num'] * (2. * np.pi / 365))
    df['day_of_year_cos'] = np.cos(df['day_num'] * (2. * np.pi / 365))
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

dfs = join_datasets([files[0]])
dfs_indices = np.random.choice(dfs.shape[0], int(dfs.shape[0]*0.2))
# dfs = dfs.sample(int(dfs.shape[0]//10))
dfs

# %%
# all_data = [X_tr, y_tr, X_v, y_v, X_test, y_test]
all_data = [dfs]
vars_to_drop = ["datetime","month", "day", "u-wind","v-wind", "week_num", "longitude", "latitude", "day_num", "hour"]
for i in range(len(all_data)):
    for var in vars_to_drop:
        try:
            all_data[i] = all_data[i].drop(var, axis=1)
        except:
            pass


# %%
dfs = all_data[0]
dfs

# %%
target = "temperature(celsius)"
data = dfs.copy()
data[f"target_{target}"] = dfs[target].shift(-6)
data = dfs.dropna()
data

# %%
X = data.drop(columns=[f'target_{target}'])  # Features DataFrame
y = data[f'target_{target}']  # Target DataFrame

# %%
from sklearn.model_selection import train_test_split

X_train_temp, X_t, y_train_temp, y_t = train_test_split(X, y, test_size=0.1, random_state=42)

X_tr, X_v, y_tr, y_v = train_test_split(X_train_temp, y_train_temp, test_size=0.1, random_state=42)


# %%
import tensorflow as tf

tf.config.experimental.list_physical_devices()
# Define your model architecture
model = tf.keras.Sequential([
    # Add an RNN layer
    tf.keras.layers.SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(None, 1)),
    tf.keras.layers.SimpleRNN(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# %%
history = model.fit(
    X_tr, 
    y_tr, 
    epochs=10,
    validation_data=(X_v, y_v)
)

# %%
model.save("6hrRNN")

predictions = model.predict(X_t)
plt.figure(figsize=(10, 6))
plt.scatter(y_t, predictions, alpha=0.5)  # Plot actual vs. predictions
plt.title('Model Predictions vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')

# Plot a diagonal line for reference
max_val = max(max(y_t), max(predictions))
min_val = min(min(y_t), min(predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # 'k--' specifies a black dashed line

plt.savefig("temp_prediction.png")


