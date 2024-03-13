% Change csv for different location, be warned this takes an age to run
data = readmatrix("10624903/-33.5N_151E.csv");
data(:,5) = data(:,5)-273.15;

% Change y and x column to customise which variable is predicted
X_before = data(:, 1:4); 
X_after = data(:, 6:end);
X = [X_before, X_after]; 

y = data(:, 5);

train_ratio = 0.8;

indices = randperm(size(data, 1));


train_size = floor(train_ratio * size(data, 1));
train_indices = indices(1:train_size);
test_indices = indices(train_size+1:end);

X_train = X(train_indices, :);
y_train = y(train_indices);

X_test = X(test_indices, :);
y_test = y(test_indices);

model = TreeBagger(numTrees, X_train, y_train, 'Method', 'regression');

y_predicted_test = predict(model, X_test);

mse_test = mean((y_test - y_predicted_test).^2);
mae_test = mean(abs(y_test - y_predicted_test));
r_squared_test = 1 - (sum((y_test - y_predicted_test).^2) / sum((y_test - mean(y_test)).^2));


fprintf('Testing Set Metrics:\n');
fprintf('Mean Squared Error (MSE): %.4f\n', mse_test);
fprintf('Mean Absolute Error (MAE): %.4f\n', mae_test);
fprintf('R-squared (R2) value: %.4f\n', r_squared_test);

