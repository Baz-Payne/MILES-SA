'''
Important note about the demand files
    - Demand files are given in half hourly amount in MW which are exclusive of hydrogen demand
'''



import pandas as pd
from datetime import datetime

# Convert half hours to 24 hour EST
half_hours = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48']
time = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', '04:00', '04:30', '05:00', '05:30', '06:00', '06:30', '07:00', '07:30', '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30']

# Create a dictionary for renaming columns
rename_dict = {old_name: new_name for old_name, new_name in zip(half_hours, time)}

# Function to create the start and end dates of a financial year
def financial_year_dates(year):
    start_date = datetime(year - 1, 7, 1)  # Financial year starts on July 1st of the previous year
    end_date = datetime(year, 6, 30, 23, 59)  # Financial year ends on June 30 of the given year
    return start_date, end_date


def update_raw_date_format(AEMO_df, data_column_name):
    '''
    Combines the separate year, month, and day columns into a single datetime column and
    then melts the separate half hour columns into a single datetime column

    Parameters
    ----------
    AEMO_df : pandas dataframe
        The original unformatted dataframe with separate year, month, day, and half hourly columns.
    
    data_column_name : string
        The type of data and unit, e.g., 'Generation (MW)'

    Returns
    -------
    updated_df : pandas dataframe
        The updated dataframe which has a single datetime column.

    '''
    
    # Merge the separate year, month, and day columns
    updated_df = AEMO_df.copy()
    dates = pd.to_datetime(updated_df[['Year', 'Month', 'Day']])
    updated_df.insert(0, 'Date', dates)
    updated_df = updated_df.drop(columns=['Year', 'Month', 'Day'])
    
    # Rename the half hourly columns to their equivalent 24-hour time value
    updated_df = updated_df.rename(columns = rename_dict)
    
    # Convert to date column to string as this makes it easy to melt with the half hours
    updated_df['Date'] = updated_df['Date'].dt.strftime("%Y-%m-%d")
    
    # Unpivot the half hourly columns
    updated_df = pd.melt(updated_df, id_vars=['Date'], var_name='Time', value_name=data_column_name)
    
    # Combine the date and time into a single datetime column
    date_times = pd.to_datetime(updated_df['Date'] + ' ' + updated_df ['Time'])
    updated_df.insert(0, 'Datetime', date_times)
    
    # Drop unnecessary columns
    updated_df = updated_df.drop(['Date', 'Time'], axis=1)
    
    # Sort by datetime
    updated_df = updated_df.sort_values(by='Datetime').reset_index(drop=True)
         
    return updated_df



### Initial Setup ###

# Generation summary file
file_path = 'Mapping Tables.xlsx'

# Specify the mapping table sheet and the hydrogen production sheet
demand_sheet_name = 'Demand Files'
regional_mapping_sheet_name = 'Regions'


# Read the in the mapping table and the hydrogen production
demand_traces = pd.read_excel(file_path, sheet_name=demand_sheet_name)
region_splits = pd.read_excel(file_path, sheet_name=regional_mapping_sheet_name)

# Get the demand file names
all_file_names = demand_traces['File Name']

# Initialise the directory that the demand files are stored in
demand_dir = 'RAW/'

# Initialise arrays to store the total demand
total_demand_no_hydrogen = []

# Use one of the files to setup and intialise the total demand to zero
total_demand_no_hydrogen = pd.read_csv(demand_dir + all_file_names[0])

# Update date format
total_demand_no_hydrogen = update_raw_date_format(total_demand_no_hydrogen, 'Demand (MW)')

# Remove additional FY2053 as it is not being considered
total_demand_no_hydrogen = total_demand_no_hydrogen[total_demand_no_hydrogen['Datetime'] < financial_year_dates(2053)[0]]

# Set all data entries to zero
total_demand_no_hydrogen.iloc[:, 1] = 0

### Total Demand Calculations Exclusive of Hydrogen Demand ###

# Go through all the demand data files
for file in all_file_names:

    # Read in the current demand
    current_demand = []
    current_demand = pd.read_csv(demand_dir + file)
    
    # Update date format
    current_demand = update_raw_date_format(current_demand, 'Demand (MW)')
    
    # Remove additional FY2053 as it is not being considered
    current_demand = current_demand[current_demand['Datetime'] < financial_year_dates(2053)[0]]
       
    # Add the current demand file to the total
    total_demand_no_hydrogen.iloc[:, 1] = total_demand_no_hydrogen.iloc[:, 1] + current_demand.iloc[:, 1]


# Save demand trace
total_demand_no_hydrogen.to_csv('Default/OperationalDemandNoH20.csv', index=False)


# Loop through all of the Honours regions and create individual traces
for region, split, file in region_splits[['Honours Region', 'Splits', 'Save File']].itertuples(index=False):
    
    regional_demand = total_demand_no_hydrogen.copy()
    regional_demand.iloc[:, 1] = regional_demand.iloc[:, 1] * split
    regional_demand.to_csv('Default/' + file, index=False)