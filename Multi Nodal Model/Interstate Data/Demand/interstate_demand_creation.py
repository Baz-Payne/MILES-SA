'''
Important notes about the demand files
    - Demand files are given in half hourly amount in MW which are exclusive of hydrogen demand
    - Hydrogen demand is given as a yearly total in TWh
    - The annual hydrogen demand is taken from AEMO electricity and gas website
    - This script splits the hydrogen demand equally and applies it to each half hour period of a given financial year
'''



import pandas as pd
from datetime import datetime


# Convert half hours to 24 hour EST
half_hours = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48']
time = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', '04:00', '04:30', '05:00', '05:30', '06:00', '06:30', '07:00', '07:30', '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30']

# Create a dictionary for renaming columns
rename_dict = {old_name: new_name for old_name, new_name in zip(half_hours, time)}


def financial_year_dates(year):
    '''
    Returns the start and end dates of a financial year

    Parameters
    ----------
    year : int
        The financial year ending, e.g., FY2024-25 ends in 2025.

    Returns
    -------
    start_date : datetime
        The first day of the financial year.
    end_date : datetime
        The last day of the financial year.

    '''
    
    # Financial year starts on July 1st of the previous year
    start_date = datetime(year - 1, 7, 1)
    
    # Financial year ends on 30th June, midnight, of the given year
    end_date = datetime(year, 6, 30, 23, 59)
    
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
NSW_demand_mapping = 'NSW Demand Files'

# Read the in the mapping table and the hydrogen production
mapping_table = pd.read_excel(file_path, sheet_name=NSW_demand_mapping)

# Get the demand file names
all_file_names = mapping_table['File Name']

# Initialise the directory that the demand files are stored in
demand_dir = 'RAW/'

# Initialise arrays to store the total demand
NSW_total_demand_no_H2 = []

# Use one of the files to setup and intialise the total demand to zero
NSW_total_demand_no_H2 = pd.read_csv(demand_dir + all_file_names[0])

# Update date format
NSW_total_demand_no_H2 = update_raw_date_format(NSW_total_demand_no_H2, 'Demand (MW)')

# Remove additional FY2053 as it is not being considered
NSW_total_demand_no_H2 = NSW_total_demand_no_H2[NSW_total_demand_no_H2['Datetime'] < financial_year_dates(2053)[0]]

# Set all data entries to zero
NSW_total_demand_no_H2.iloc[:, 1:] = 0


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
    NSW_total_demand_no_H2.iloc[:, 1:] = NSW_total_demand_no_H2.iloc[:, 1:] + current_demand.iloc[:, 1:]


# Save demand trace
NSW_total_demand_no_H2.to_csv('Default/NSWStepChangeDemandNoH2.csv', index=False)
      
# Initialise arrays to store the total demand
VIC_total_demand_no_H2 = []
VIC_total_demand_no_H2 = pd.read_csv(demand_dir + 'VIC_RefYear_4006_STEP_CHANGE_POE10_OPSO_MODELLING.csv')

# Update date format
VIC_total_demand_no_H2 = update_raw_date_format(VIC_total_demand_no_H2, 'Demand (MW)')
VIC_total_demand_no_H2 = VIC_total_demand_no_H2[VIC_total_demand_no_H2['Datetime'] < financial_year_dates(2053)[0]]
VIC_total_demand_no_H2.to_csv('Default/VICStepChangeDemandNoH2.csv', index=False)


# Create empty dataframes with the net zero years as the column headings for input totals
NSW_annual_demand_no_H2 = pd.DataFrame(columns=list(range(2025, 2052)))
VIC_annual_demand_no_H2 = pd.DataFrame(columns=list(range(2025, 2052)))

# Loop through each financial year and add hydrogen to the entries.
for year in range(2025, 2053):
    
    start_date, end_date = financial_year_dates(year)

     
    # Add up the demand in each FY both with and without hydrogen
    demand_FY_NSW_no_H2 = NSW_total_demand_no_H2.loc[(NSW_total_demand_no_H2['Datetime'] >= start_date) & (NSW_total_demand_no_H2['Datetime'] <= end_date), NSW_total_demand_no_H2.columns[1:]]
    demand_FY_VIC_no_H2 = VIC_total_demand_no_H2.loc[(VIC_total_demand_no_H2['Datetime'] >= start_date) & (VIC_total_demand_no_H2['Datetime'] <= end_date), VIC_total_demand_no_H2.columns[1:]]

    # Multiply by 0.5 here as we are adding over half hour intervals not hours
    NSW_FY_total = 0.5 * demand_FY_NSW_no_H2.iloc[:, :].sum()
    VIC_FY_total = 0.5 * demand_FY_VIC_no_H2.iloc[:, :].sum()
    
    # Divide to convert to TWh
    NSW_annual_demand_no_H2.loc[0, year] = NSW_FY_total.sum(axis=0) / 10**6 
    VIC_annual_demand_no_H2.loc[0, year] = VIC_FY_total.sum(axis=0) / 10**6 


