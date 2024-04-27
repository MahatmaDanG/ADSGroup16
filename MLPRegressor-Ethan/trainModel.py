# %%
#So I want to use scikit learn's MLP regressor model to predict temperature 
import pandas as pd
from IPython.display import display
# Define column names
columns = ['year', 'month', 'day', 'hour', 'temperature', 'precipitation', 'u-wind', 'v-wind']

# Load data from CSV without headers
df = pd.read_csv("17.36N_78.5E.csv", names=columns) #Here we are only choosing one of the CSV files, 
                                                            #this can be expanded to make use of the full data set
df['temperature'] -= 273.15
#Now I have the dataset loaded and the temperature adjusted to celcius
df

# %%
#This cell just trims the entire dataframe down to make testing faster, here we take the first 20% of the rows...
import numpy as np
percentage = 0.4  # Top 10% of dataframe rows...
df_top = df[(df.index > np.percentile(df.index, 100-(percentage*10)))]
num_rows = len(df)
top_rows = int(num_rows * percentage)
df_top = df.head(top_rows)
df = df_top
df

# %%
import pandas as pd
#This cell produces new one-hot encoded values from the date columns, classifying them as seasons, times of day
#We also make the day of week cyclic, classifying them to monday,tuesday,wednesday... as numbers 0-6

print(df['hour'].unique())
# Assuming df is your dataframe
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df['season'] = df['datetime'].dt.month // 3 + 1  # Divide months into seasons
df['time_of_day'] = pd.cut(df['hour'], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]) # 0,1,2,3 represent night,morning,afternoon,evening
df['day_of_week'] = df['datetime'].dt.day_of_week
df['time_since_start'] = (df['datetime'] - df['datetime'].min()).dt.days
df.drop(columns=['datetime'], inplace=True)


# %%
#Optionally perform PCA dimensionality reduction or other preprocessing here...

# %%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['temperature']),  # Features (X)
    df['temperature'],  # Target variable (y)
    test_size=0.2,  # 20% for test set
    random_state=1  # For reproducibility
)

# Further split the train set into train and validation sets (60% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,  # 20% for validation set (0.25 * 0.8 = 0.2)
    random_state=1  # For reproducibility
)

# Now you have X_train, y_train (for training),
# X_val, y_val (for validation), and
# X_test, y_test (for testing)
display(X_train) # X_train contains our data values excluding temperature
display(y_train) # y_train is our target variable dataframe

# %%
#Check for not a number values:
nan_values = df.isna()
print(nan_values.sum())
display(df['time_of_day'])
nan_values = df['time_of_day'].isna()
print(nan_values)
display(df['time_of_day'].unique())

# %%
# Create an MLPRegressor with one hidden layer of 100 neurons
model = MLPRegressor(hidden_layer_sizes=(128, 64, 32, 16), activation='relu', solver='adam', max_iter=250, verbose=True)

# Fit the model to your training data
model.fit(X_train, y_train)


# %%
8092375 <------ THIS IS MY LATEST TENSORFLOW TRAINING JOB

# %%
#Predicting values over my test set...
y_pred = model.predict(X_test)

# %%
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


# %%
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residuals':(abs(y_test-y_pred))})
display(df_results.head())
print()
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("The accuracy of our model is {}%".format(round(score, 5) *100))

# # %%
# import matplotlib.pyplot as plt
# test_loss = mean_squared_error(y_test, y_pred)

# # Plot test loss
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, color='blue', label='Test data', s = 5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect prediction')
# plt.title('Test Loss (MSE): {:.2f}'.format(test_loss))
# plt.xlabel('True values')
# plt.ylabel('Predicted values')
# plt.legend()
# plt.grid(True)
# plt.show()


