import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Kaggle dataset
data = pd.read_csv("train.csv")

print("Dataset loaded successfully")
print("\nFirst 5 rows:")
print(data.head())


# Select required columns
# GrLivArea = Square footage
# BedroomAbvGr = Bedrooms
# FullBath = Bathrooms
# SalePrice = House price

X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]

print("\nSelected input columns:")
print(X.head())

print("\nOutput column:")
print(y.head())


# Check missing values
print("\nMissing values:")
print(X.isnull().sum())


# Fill missing values if any
X = X.fillna(X.mean())


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create Linear Regression model using 3 features
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Model Evaluation
print("\nModel Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


# Take input from user
print("\nEnter New House Details")

square_footage = float(input("Enter Square Footage: "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))


# Predict new house price using 3-feature model
new_house = pd.DataFrame({
    "GrLivArea": [square_footage],
    "BedroomAbvGr": [bedrooms],
    "FullBath": [bathrooms]
})

predicted_price = model.predict(new_house)
predicted_value = round(predicted_price[0], 2)

print("\nNew House Details:")
print("Square Footage:", square_footage)
print("Bedrooms:", bedrooms)
print("Bathrooms:", bathrooms)
print("Predicted House Price:", predicted_value)


# ---------------------------------------------------
# Single Graph: Dynamic Zoomed Regression View
# ---------------------------------------------------

X_graph = data[["GrLivArea"]]
y_graph = data["SalePrice"]

# Create separate simple linear regression model for straight graph line
graph_model = LinearRegression()
graph_model.fit(X_graph, y_graph)

# Create graph range near user input
min_sqft = square_footage - 700
max_sqft = square_footage + 700

square_feet_range = np.linspace(min_sqft, max_sqft, 100).reshape(-1, 1)

# Predict prices for straight regression line
regression_line = graph_model.predict(square_feet_range)

# Select nearby actual house prices
nearby_data = data[
    (data["GrLivArea"] >= min_sqft) &
    (data["GrLivArea"] <= max_sqft)
]

plt.figure(figsize=(10, 6))

# Nearby actual house price points
plt.scatter(
    nearby_data["GrLivArea"],
    nearby_data["SalePrice"],
    color="green",
    label="Actual House Prices"
)

# Straight regression line
plt.plot(
    square_feet_range,
    regression_line,
    color="red",
    linewidth=2,
    label="Regression Line"
)

# User input predicted point
plt.scatter(
    square_footage,
    predicted_value,
    color="blue",
    s=250,
    edgecolors="black",
    label="Your Predicted Price"
)

# Vertical line for user square footage
plt.axvline(
    x=square_footage,
    color="blue",
    linestyle="--",
    label="Your Square Footage"
)

# Horizontal line for predicted price
plt.axhline(
    y=predicted_value,
    color="purple",
    linestyle="--",
    label="Predicted Price Line"
)

# Show predicted value on graph
plt.annotate(
    f"Predicted Price: {predicted_value}\nSqft: {square_footage}\nBedrooms: {bedrooms}\nBathrooms: {bathrooms}",
    xy=(square_footage, predicted_value),
    xytext=(square_footage + 100, predicted_value + 20000),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white")
)

plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.title(f"House Price Prediction for {square_footage} Sqft")
plt.legend()
plt.grid(True)

# Dynamic graph limits
plt.xlim(min_sqft, max_sqft)
plt.ylim(predicted_value - 100000, predicted_value + 100000)

plt.show()