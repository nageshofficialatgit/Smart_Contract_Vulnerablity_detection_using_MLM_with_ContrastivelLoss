import pandas as pd

# Load the CSV file
csv_file = 'Nagesh.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Define the vulnerability columns
vulnerability_columns = [
    'Reentrancy', 'Integer_Overflow_Underflow', 'Access_Control', 
    'Timestamp_Dependence', 'Transaction_Ordering_Dependence_Front_Running', 
    'Bad_Randomness', 'Unchecked_Low_Level_Calls'
]

# Add a "Vulnerable" column based on whether any vulnerability column is non-zero
df['Vulnerable'] = df[vulnerability_columns].apply(lambda x: '1' if any(x > 0) else '0', axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv('Labelled_ICSE.csv', index=False)
