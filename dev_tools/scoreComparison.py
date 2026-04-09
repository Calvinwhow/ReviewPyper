import pandas as pd
import numpy as np

# --- Configuration: Customize These Values ---
# 1. Map column indices to categories 'a', 'b', 'c', 'd'.
#    Replace the placeholders (e.g., 4, 7, 10, 13) with the actual 
#    column indices from the CSV you want to include in the calculation.
column_map = {
    'a': [4, 6, 9],  # Example indices for category 'a'
    'b': [12, 15],  # Example indices for category 'b'
    'c': [], # Example indices for category 'c'
    'd': []  # Example indices for category 'd'
}

# 2. Define the multiplier for each category.
#    I've used the example values you provided: a=2, b=7, c=8, d=4.
multipliers = {
    'a': 2,
    'b': 7,
    'c': 8,
    'd': 4
}
# ---------------------------------------------

# Load the CSV file
df = pd.read_csv('master_list.csv')

# --- Data Cleaning and Filtering Function ---
def process_column(series):
    """
    Cleans the column:
    1. Converts column to numeric, coercing non-numeric values to NaN.
    2. Replaces values greater than 4 with NaN (as per request).
    3. Converts remaining NaN to 0 for summation.
    """
    # 1. Convert to numeric, errors='coerce' turns non-convertible text into NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # 2. Filter: If the value is > 4, do not parse it (set to NaN)
    # The condition is: value > 4. We use .mask() to set these values to NaN.
    filtered_series = numeric_series.mask(numeric_series > 4, other=np.nan)
    
    # 3. Fill remaining NaN values with 0 for summation
    return filtered_series.fillna(0)

# --- Calculation ---
final_sum_series = pd.Series(0, index=df.index)

# Iterate through each category and its columns
for category, indices in column_map.items():
    multiplier = multipliers[category]
    
    # Process each column in the current category
    for index in indices:
        # Get the column by index (df.columns[index] is the column name)
        # Use .iloc[:, index] to select the data by integer position
        column_data = df.iloc[:, index]
        
        # Clean, filter, and fill the column data
        processed_values = process_column(column_data)
        
        # Scale the processed column by the category multiplier
        scaled_column = processed_values * multiplier
        
        # Add the scaled column values to the final running sum
        final_sum_series = final_sum_series + scaled_column

# --- Output ---
# Display the first 5 calculated sums (one per row)
print("--- First 5 Final Calculated Sums (One per Row) ---")
print(final_sum_series.head().to_markdown(numalign='left', stralign='left'))

# Display the total sum of all final calculated row sums
total_sum = final_sum_series.sum()
print(f"\nTotal Sum of all scaled, filtered column values (sum of all rows): {total_sum}")

# Save the final calculated sum column back to the DataFrame
df['Final_Scaled_Sum'] = final_sum_series
# df.to_csv('master_list_with_sum.csv', index=False) # Uncomment to save the file
