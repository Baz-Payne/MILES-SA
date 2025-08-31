'''
Important notes about the demand and generation files
    - Demand files are given in MW
    - Generation files are given in MW
'''


import pandas as pd
import numpy as np
from datetime import datetime


# Function to create the start and end points of a financial year
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



def calendar_year_dates(year):
    start_date = datetime(year, 1, 1)  # Calendar year starts on Jan 1st
    end_date = datetime(year, 12, 31, 23, 59, 59)  # Calendar year ends on Dec 31st
    return start_date, end_date


### Read in Demand and Generation Files ###

# Storage summary file
file_path = 'Mapping Tables.xlsx'

# Specify the mapping table sheet
NSW_sheet_name_mapping = 'NSW Files'
VIC_sheet_name_mapping = 'VIC Files'
NSW_sheet_name_H2 = 'NSW H2 Demand'
VIC_sheet_name_H2 = 'VIC H2 Demand'

# Read the in the mapping table and the hydrogen production
NSW_mapping_table = pd.read_excel(file_path, sheet_name=NSW_sheet_name_mapping)
VIC_mapping_table = pd.read_excel(file_path, sheet_name=VIC_sheet_name_mapping)
NSW_hydrogen_demand = pd.read_excel(file_path, sheet_name=NSW_sheet_name_H2)
VIC_hydrogen_demand = pd.read_excel(file_path, sheet_name=VIC_sheet_name_H2)


# Get the demand file names
NSW_file_names = NSW_mapping_table['File Name']
VIC_file_names = VIC_mapping_table['File Name']

# Initialise the directory that the demand files are stored in
demand_dir = 'Demand/Default/'
solar_dir = 'Generation/Default/'
wind_dir = 'Generation/Default/'


# Initialise array to store the total net demand
NSW_total_net_demand = []
VIC_total_net_demand = []

# Read in the demand, solar and wind
NSW_demand = pd.read_csv(demand_dir + NSW_mapping_table.loc[NSW_mapping_table['Type'] == 'NSW Demand', 'File Name'].iloc[0])
NSW_demand['Datetime'] = pd.to_datetime(NSW_demand['Datetime'])
NSW_solar = pd.read_csv(solar_dir + NSW_mapping_table.loc[NSW_mapping_table['Type'] == 'NSW Solar', 'File Name'].iloc[0])
NSW_solar['Datetime'] = pd.to_datetime(NSW_solar['Datetime'])
NSW_wind = pd.read_csv(wind_dir + NSW_mapping_table.loc[NSW_mapping_table['Type'] == 'NSW Wind', 'File Name'].iloc[0])
NSW_wind['Datetime'] = pd.to_datetime(NSW_wind['Datetime'])


VIC_demand = pd.read_csv(demand_dir + VIC_mapping_table.loc[VIC_mapping_table['Type'] == 'VIC Demand', 'File Name'].iloc[0])
VIC_demand['Datetime'] = pd.to_datetime(VIC_demand['Datetime'])
VIC_solar = pd.read_csv(solar_dir + VIC_mapping_table.loc[VIC_mapping_table['Type'] == 'VIC Solar', 'File Name'].iloc[0])
VIC_solar['Datetime'] = pd.to_datetime(VIC_solar['Datetime'])
VIC_wind = pd.read_csv(wind_dir + VIC_mapping_table.loc[VIC_mapping_table['Type'] == 'VIC Wind', 'File Name'].iloc[0])
VIC_wind['Datetime'] = pd.to_datetime(VIC_wind['Datetime'])

# Calculate the total net demand with hydrogen (operational demand (no hydrogen) - solar - wind). We ignore the first column as they are the dates
NSW_total_net_demand = NSW_demand.copy()
NSW_total_net_demand.iloc[:, 1:] = NSW_total_net_demand.iloc[:, 1] - NSW_solar.iloc[:, 1] - NSW_wind.iloc[:, 1]

VIC_total_net_demand = VIC_demand.copy()
VIC_total_net_demand.iloc[:, 1:] = VIC_total_net_demand.iloc[:, 1] - VIC_solar.iloc[:, 1] - VIC_wind.iloc[:, 1]


# Create a hydrogen demandtrace file so the generation can be added to figures
NSW_H2_demand_trace = NSW_wind.copy()
NSW_H2_demand_trace.iloc[:,1] = 0
VIC_H2_demand_trace = NSW_H2_demand_trace.copy()


# We first need to add hydrogen demand as this should be included at time points when demand is negative after renewables have been considered
for year in range(2025, 2053):
    
    # Obtain the start and end dates of the financial year
    start_date, end_date = financial_year_dates(year)
    
    # Extract the net demand in the current financial year
    NSW_demand_FY = NSW_total_net_demand.loc[(NSW_total_net_demand['Datetime'] >= start_date) & (NSW_total_net_demand['Datetime'] <= end_date), NSW_total_net_demand.columns[:]]
    VIC_demand_FY = VIC_total_net_demand.loc[(VIC_total_net_demand['Datetime'] >= start_date) & (VIC_total_net_demand['Datetime'] <= end_date), VIC_total_net_demand.columns[:]]
    
    # Count the number of half hour periods the net demand is negative
    NSW_negative_demand_count = (NSW_demand_FY['Demand (MW)'] < 0).sum().sum()
    VIC_negative_demand_count = (VIC_demand_FY['Demand (MW)'] < 0).sum().sum()
    
    # Calculate the hydrogen demand to be added in each negative demand period. Times by 2 to convert to hh. Times 10^6 to convert from TWh to MW
    NSW_max_hydrogen_demand = 2 * NSW_hydrogen_demand.at[0, year] * 10**6 / NSW_negative_demand_count
    VIC_max_hydrogen_demand = 2 * VIC_hydrogen_demand.at[0, year] * 10**6 / VIC_negative_demand_count

    
    # Determine the indices to add hydrogen demand
    NSW_filtered_indices = np.array(NSW_demand_FY['Demand (MW)'] < 0)
    VIC_filtered_indices = np.array(VIC_demand_FY['Demand (MW)'] < 0)
    
    # Create the FY hydrogen trace and then update the forecast period dataframe
    NSW_hydrogen_FY = NSW_H2_demand_trace.loc[(NSW_H2_demand_trace['Datetime'] >= start_date) & (NSW_H2_demand_trace['Datetime'] <= end_date), NSW_H2_demand_trace.columns[:]]
    NSW_hydrogen_FY.loc[NSW_filtered_indices, 'Generation (MW)'] += NSW_max_hydrogen_demand
    NSW_H2_demand_trace.loc[(NSW_H2_demand_trace['Datetime'] >= start_date) & (NSW_H2_demand_trace['Datetime'] <= end_date), NSW_H2_demand_trace.columns[:]] = NSW_hydrogen_FY

    VIC_hydrogen_FY = VIC_H2_demand_trace.loc[(VIC_H2_demand_trace['Datetime'] >= start_date) & (VIC_H2_demand_trace['Datetime'] <= end_date), VIC_H2_demand_trace.columns[:]]
    VIC_hydrogen_FY.loc[VIC_filtered_indices, 'Generation (MW)'] += VIC_max_hydrogen_demand
    VIC_H2_demand_trace.loc[(VIC_H2_demand_trace['Datetime'] >= start_date) & (VIC_H2_demand_trace['Datetime'] <= end_date), VIC_H2_demand_trace.columns[:]] = VIC_hydrogen_FY


######
# Save the total net demand dataframe for future reference
#######

NSW_demand.iloc[:,1] = NSW_demand.iloc[:,1] + NSW_H2_demand_trace.iloc[:,1]
VIC_demand.iloc[:,1] = VIC_demand.iloc[:,1] + VIC_H2_demand_trace.iloc[:,1]


# Save the hydrogen demand trace
NSW_H2_demand_trace.to_csv('H2 Traces/NSWHydrogenDemandTrace.csv', index=False)
VIC_H2_demand_trace.to_csv('H2 Traces/VICHydrogenDemandTrace.csv', index=False)

NSW_demand.to_csv('Final Traces/Operational Demand/NSWStepChangeOperationalDemand.csv', index=False)
VIC_demand.to_csv('Final Traces/Operational Demand/VICStepChangeOperationalDemand.csv', index=False)


# Save the net demand traces to be used in the multi nodal model
NSW_net = NSW_demand.copy()
NSW_net.iloc[:, 1:] = NSW_demand.iloc[:, 1] - NSW_solar.iloc[:, 1] - NSW_wind.iloc[:, 1]
NSW_net.to_csv('Final Traces/NSWNetDemand.csv', index=False)

VIC_net = VIC_demand.copy()
VIC_net.iloc[:, 1:] = VIC_demand.iloc[:, 1] - VIC_solar.iloc[:, 1] - VIC_wind.iloc[:, 1]
VIC_net.to_csv('Final Traces/VICNetDemand.csv', index=False)


# Sanity check
NSW_annual_demand = pd.DataFrame(columns=list(range(2025, 2052)))
VIC_annual_demand = pd.DataFrame(columns=list(range(2025, 2052)))

for year in range(2025, 2053):
    
    start_date, end_date = financial_year_dates(year)

     
    # Add up the demand in each FY both with and without hydrogen
    demand_FY_NSW = NSW_demand.loc[(NSW_demand['Datetime'] >= start_date) & (NSW_demand['Datetime'] <= end_date), NSW_demand.columns[1:]]
    demand_FY_VIC = VIC_demand.loc[(NSW_demand['Datetime'] >= start_date) & (NSW_demand['Datetime'] <= end_date), NSW_demand.columns[1:]]

    # Multiply by 0.5 here as we are adding over half hour intervals not hours
    NSW_FY_total = 0.5 * demand_FY_NSW.iloc[:, :].sum()
    VIC_FY_total = 0.5 * demand_FY_VIC.iloc[:, :].sum()
    
    # Divide to convert to TWh
    NSW_annual_demand.loc[0, year] = NSW_FY_total.sum(axis=0) / 10**6 
    VIC_annual_demand.loc[0, year] = VIC_FY_total.sum(axis=0) / 10**6 