import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Path of the file to read
house_price_data = "train.csv"

# Read the file into a variable home_data
home_data = pd.read_csv(house_price_data)

print(home_data)

# Print summary statistics in next line
print(home_data.describe())

print(home_data.columns.values)

# Find the average lot size (rounded to nearest integer)
avg_lot_size = home_data.loc[:,"LotArea"].mean()
print(int(avg_lot_size))

# Find the newest home (current year - the date in which it was built)
newest_home_age = home_data.loc[:,"YearBuilt"].max()
print(newest_home_age)

# Find the latest sale
latest_sale = home_data.loc[:,"YrSold"].max()
print(latest_sale)

# Define the target variable, sales price
y = home_data["SalePrice"]

# Define the predictive features
house_features=["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = home_data[house_features]

# Check the predictive features data
print(x.describe())
print(x)

# Split data into training and validation data, for both features and target
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

# Define model. Specify a number for random_state to ensure same results each run
house_model = DecisionTreeRegressor(random_state=1)

# Fit model
house_model.fit(train_x, train_y)

# Make predictions with the model's predict command using x as the data. Save the results to a variable called predictions.
preds_val = house_model.predict(val_x)
print(preds_val)

# Compare the predictions with the actual home values by looking at the first few results

print(preds_val[:4])
print(val_y.head())
print(mean_absolute_error(val_y, preds_val))

# Define function that retrieves the mean absolute error based on different numbers of maximum leaf nodes
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Create a dictionary containing a range of leaf node numbers with their mae values
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

mae_values = {}

for node in candidate_max_leaf_nodes:
    mae = get_mae(node, train_x, val_x, train_y, val_y)
    mae_values[node] = mae

# Retrieve the optimum tree size with the lowest mae
best_tree_size = min(mae_values, key=mae_values.get)

# Check the dictionary and best tree size numbers
print(mae_values)
print(best_tree_size)

# Define the refined model
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 0)

# Now that model has been refined, use all the data to fit it
final_model.fit(x, y)

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state = 1)

# Fit model
rf_model.fit(train_x, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_predictions = rf_model.predict(val_x)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print(f"Validation MAE for Random Forest Model: {rf_val_mae}")

plt.scatter(home_data["LotArea"], home_data["SalePrice"])
plt.show()