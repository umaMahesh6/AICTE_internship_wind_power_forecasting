# Import necessary libraries
import pandas as pd  # Data manipulation
import numpy as np  # Numerical computations
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Advanced visualization

# Load the CSV (Dataset) files
location1 = pd.read_csv(r'C:\Users\saima\PycharmProjects\pythonProject1\AICTE_WIND_POWER_GENERATION_FORECASTING\Location1.csv')
location2 = pd.read_csv(r'C:\Users\saima\PycharmProjects\pythonProject1\AICTE_WIND_POWER_GENERATION_FORECASTING\Location2.csv')
location3 = pd.read_csv(r'C:\Users\saima\PycharmProjects\pythonProject1\AICTE_WIND_POWER_GENERATION_FORECASTING\Location3.csv')
location4 = pd.read_csv(r'C:\Users\saima\PycharmProjects\pythonProject1\AICTE_WIND_POWER_GENERATION_FORECASTING\Location4.csv')

# Display the first 5 rows of each dataset
print("\n\nHead of all dataset:\n")
print(location1.head(), "\n")
print(location2.head(), "\n")
print(location3.head(), "\n")
print(location4.head(), "\n")

print("\n\nTail of all datasets:")
print(location1.tail(), "\n")
print(location2.tail(), "\n")
print(location3.tail(), "\n")
print(location4.tail(), "\n")

# Get summary statistics of each dataset
print("\n\nSummary statistics of all dataset:\n\n")
print(location1.describe(), "\n")
print(location2.describe(), "\n")
print(location3.describe(), "\n")
print(location4.describe(), "\n")

print("\n\nMissing values in all dataset:\n\n")
print(location1.isnull().sum(), "\n")
print(location2.isnull().sum(), "\n")
print(location3.isnull().sum(), "\n")
print(location4.isnull().sum(), "\n")

# Display the column names
print("\n\nColumn names in Location1 dataset:", location1.columns, "\n")
print("Column names in Location2 dataset:", location2.columns, "\n")
print("Column names in Location3 dataset:", location3.columns, "\n")
print("Column names in Location4 dataset:", location4.columns, "\n")

# Get the number of rows and columns in each dataset
print("\n\nShape of Location1 dataset (rows, columns):", location1.shape, "\n")
print("Shape of Location2 dataset (rows, columns):", location2.shape, "\n")
print("Shape of Location3 dataset (rows, columns):", location3.shape, "\n")
print("Shape of Location4 dataset (rows, columns):", location4.shape, "\n")

print("\n\nData types in all datasets:")
print(location1.dtypes, "\n")
print(location2.dtypes, "\n")
print(location3.dtypes, "\n")
print(location4.dtypes, "\n")

# Compute the mean of each numerical column
print("\n\nMean values in all datasets:\n")
print(location1.mean(numeric_only=True), "\n")
print(location2.mean(numeric_only=True), "\n")
print(location3.mean(numeric_only=True), "\n")
print(location4.mean(numeric_only=True), "\n")

print("\n\nMedian values in all datasets:")
print(location1.median(numeric_only=True), "\n")
print(location2.median(numeric_only=True), "\n")
print(location3.median(numeric_only=True), "\n")
print(location4.median(numeric_only=True), "\n")

print("\n\nInfo of Location1 dataset:")
location1.info()
print("\n")
print("Info of Location2 dataset:")
location2.info()
print("\n")
print("Info of Location3 dataset:")
location3.info()
print("\n")
print("Info of Location4 dataset:")
location4.info()
print("\n")
merged_data = pd.concat([location1, location2, location3, location4], ignore_index=True)

print("\n\nMerged Data - First 5 Rows:")
print(merged_data.head(), "\n")
print("\nMerged Data - Info:")
merged_data.info()
print("\n")
print("\nMerged Data - Statistical Summary:")
print(merged_data.describe(), "\n")
# Check for missing values
print("\n\nMissing Values in Merged Data:")
print(merged_data.isnull().sum(), "\n")
# Check for duplicated rows
print("\n\nNumber of Duplicated Rows:")
print(merged_data.duplicated().sum(), "\n")
# Convert categorical column 'Location' into dummy variables (if exists)
if 'Location' in merged_data.columns:
    print("\n\nPerforming One-Hot Encoding on 'Location' Column...\n")
    merged_data = pd.get_dummies(merged_data, columns=['Location'], drop_first=True)
    print("\nMerged Data - First 5 Rows After Encoding:")
    print(merged_data.head(), "\n")

# Display final column names
print("\n\nColumn Names in Merged Data:")
print(merged_data.columns, "\n")

# Drop the 'Time' column (if exists)
if 'Time' in merged_data.columns:
    print("\n\nDropping 'Time' Column...\n")
    merged_data.drop('Time', axis=1, inplace=True)
    print("\nMerged Data - First 5 Rows After Dropping 'Time' Column:")
    print(merged_data.head(), "\n")

plt.figure(figsize=(8, 5))
sns.histplot(merged_data['Power'], bins=25, kde=True, color='green')
plt.xlabel('Power (kW)')
plt.ylabel('Frequency')
plt.title('Distribution of Wind Power Generation')
plt.show()


