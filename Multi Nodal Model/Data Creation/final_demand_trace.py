'''
Important notes about the demand and generation files
    - Demand files are given in MW
    - Generation files are given in MW
'''


import pandas as pd
import numpy as np
from datetime import datetime


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



### Read in Demand and Generation Files ###

# Storage summary file
file_path = 'Mapping Tables.xlsx'

# Specify the mapping table sheet
sheet_name_mapping = 'Gen and Demand Files'
sheet_name_hydrogen = 'Hydrogen Demand'

# Read the in the mapping table and the hydrogen production
mapping_table = pd.read_excel(file_path, sheet_name=sheet_name_mapping)
hydrogen_demand = pd.read_excel(file_path, sheet_name=sheet_name_hydrogen)

# Get the demand file names
all_file_names = mapping_table['File Name']

# Initialise the directory that the demand files are stored in
demand_dir = 'Demand/Default/'
gen_dir = 'Generation/Default/'


# Initialise array to store the total net demand
total_net_demand = []


# Read in the demand files and generation files
demand = pd.read_csv(demand_dir + mapping_table.loc[mapping_table['Type'] == 'Demand No Hydrogen', 'File Name'].iloc[0])
demand['Datetime'] = pd.to_datetime(demand['Datetime'])
eyre_demand_no_h20 = pd.read_csv(demand_dir + mapping_table.loc[mapping_table['Type'] == 'Eyre Peninsula', 'File Name'].iloc[0])
eyre_demand_no_h20['Datetime'] = pd.to_datetime(eyre_demand_no_h20['Datetime'])
north_demand_no_h20 = pd.read_csv(demand_dir + mapping_table.loc[mapping_table['Type'] == 'The North', 'File Name'].iloc[0])
north_demand_no_h20['Datetime'] = pd.to_datetime(north_demand_no_h20['Datetime'])
generation = pd.read_csv(gen_dir + mapping_table.loc[mapping_table['Type'] == 'Generation', 'File Name'].iloc[0])
generation['Datetime'] = pd.to_datetime(generation['Datetime'])


# Calculate the total net demand with hydrogen (operational demand (no hydrogen) - generation). We ignore the first column as they are the dates
total_net_demand = demand.copy()
total_net_demand.iloc[:, 1:] = total_net_demand.iloc[:, 1] - generation.iloc[:, 1]

# Create a hydrogen demandtrace file so the generation can be added to figures
hydrogen_demand_trace = demand.copy()
hydrogen_demand_trace.iloc[:,1] = 0


# We first need to add hydrogen demand as this should be included at time points when demand is negative after renewables have been considered
for year in range(2025, 2053):
    
    # Obtain the start and end dates of the financial year
    start_date, end_date = financial_year_dates(year)
    
    # Extract the net demand in the current financial year
    demand_FY = total_net_demand.loc[(total_net_demand['Datetime'] >= start_date) & (total_net_demand['Datetime'] <= end_date), total_net_demand.columns[:]]
    
    # Count the number of half hour periods the net demand is negative
    negative_demand_count = (demand_FY['Demand (MW)'] < 0).sum().sum()
    
    # Calculate the hydrogen demand to be added in each negative demand period. Times by 2 to convert to hh. Times 10^6 to convert from TWh to MW
    max_hydrogen_demand = 2 * hydrogen_demand.at[0, year] * 10**6 / negative_demand_count

    """
    print(max_hydrogen_demand)
    print(negative_demand_count)
    print('SUM: ', max_hydrogen_demand * negative_demand_count * 10 ** (-6) / 2)
    """
    
    # Determine the indices to add hydrogen demand
    filtered_indices = np.array(demand_FY['Demand (MW)'] < 0)
    
    # Create the FY hydrogen trace and then update the forecast period dataframe
    hydrogen_FY = hydrogen_demand_trace.loc[(hydrogen_demand_trace['Datetime'] >= start_date) & (hydrogen_demand_trace['Datetime'] <= end_date), hydrogen_demand_trace.columns[:]]
    hydrogen_FY.loc[filtered_indices, 'Demand (MW)'] += max_hydrogen_demand
    hydrogen_demand_trace.loc[(hydrogen_demand_trace['Datetime'] >= start_date) & (hydrogen_demand_trace['Datetime'] <= end_date), hydrogen_demand_trace.columns[:]] = hydrogen_FY


######
# Save the total net demand dataframe for future reference
#######

# Update the demand with the hydrogen load trace
demand.iloc[:,1] += hydrogen_demand_trace.iloc[:,1]

# Add the hydrogen demand to the two Honours regions
eyre_demand_no_h20.iloc[:,1] += 0.5 * hydrogen_demand_trace.iloc[:,1]
north_demand_no_h20.iloc[:,1] += 0.5 * hydrogen_demand_trace.iloc[:,1]

# Save the traces
hydrogen_demand_trace.to_csv('Final Traces/HydrogenDemandTrace.csv', index=False)
demand.to_csv('Final Traces/StepChangeOperationalDemand.csv', index=False)

eyre_demand_no_h20.to_csv('Final Traces/EyrePeninsulaDemand.csv', index=False)
north_demand_no_h20.to_csv('Final Traces/TheNorthDemand.csv', index=False)